import numpy as np
import torch
import torch.nn as nn
import os
import csv
import scipy.interpolate
from autoware_vehicle_adaptor.training.early_stopping import EarlyStopping
from autoware_vehicle_adaptor.training import convert_model_to_csv
from autoware_vehicle_adaptor.param import parameters
from scipy.ndimage import gaussian_filter
import copy

class InputsSchedulePredictorNN(nn.Module):
    def __init__(self, dataset_num, lstm_hidden_size=64,
                 post_decoder_hidden_size=(16,16),control_dt=0.033,target_size=35, input_rate_limit=1.0,
                 augmented_input_size=15, vel_scaling=0.1, vel_bias=0.0):#,vel_bias=5.0):
        super(InputsSchedulePredictorNN, self).__init__()
        self.dataset_num = dataset_num
        self.control_dt = control_dt
        self.target_size = target_size
        self.input_size = 2
        self.output_size = 1
        self.augmented_input_size = augmented_input_size
        self.vel_scaling = vel_scaling
        self.vel_bias = vel_bias
        self.lstm_encoder = nn.LSTM(self.input_size, lstm_hidden_size, num_layers=2, batch_first=True)
        self.lstm_decoder = nn.LSTM(self.output_size + self.augmented_input_size, lstm_hidden_size, num_layers=2, batch_first=True)
        self.post_decoder = nn.Sequential(
            nn.Linear(lstm_hidden_size, post_decoder_hidden_size[0]),
            nn.ReLU()
        )
        self.post_decoder_adaptive_scales = nn.ParameterList([nn.Parameter(torch.ones(post_decoder_hidden_size[0])) for _ in range(dataset_num)])
        self.finalize = nn.Sequential(
            nn.Linear(post_decoder_hidden_size[0], post_decoder_hidden_size[1]),
            nn.ReLU(),
            nn.Linear(post_decoder_hidden_size[1], self.output_size),
            nn.Tanh()
        )

        input_transition_matrix = torch.zeros(self.augmented_input_size, self.augmented_input_size)
        input_transition_matrix[1:, :-1] = torch.eye(self.augmented_input_size - 1)
        input_transition_matrix[-1, -1] = 1.0
        add_new_input = torch.zeros(1,self.augmented_input_size)
        add_new_input[0,-1] = 1.0
        self.register_buffer("input_transition_matrix", input_transition_matrix)
        self.register_buffer("add_new_input", add_new_input)
        self.input_rate_limit = input_rate_limit
    def forward(self, x, dataset_idx, true_prediction=None):
        device = x.device
        x_vel_scaled = x.clone()
        x_vel_scaled[:,:,0] = (x_vel_scaled[:,:,0] - self.vel_bias ) * self.vel_scaling
        #x_vel_scaled = self.pre_encoder(x_vel_scaled)
        post_decoder_scales_tensor = torch.stack(list(self.post_decoder_adaptive_scales)).to(device)
        post_decoder_scales = post_decoder_scales_tensor[dataset_idx]
        post_decoder_scales = post_decoder_scales.unsqueeze(1)
        _, (hidden_raw, cell_raw) = self.lstm_encoder(x_vel_scaled)
        #_, (hidden_mean, cell_mean) = self.lstm_encoder_mean(x_vel_scaled_mean)
        hidden = hidden_raw#torch.cat([hidden_raw, hidden_mean], dim=2)
        cell = cell_raw#torch.cat([cell_raw, cell_mean], dim=2)
        #hidden = torch.cat([hidden_raw, hidden_mean], dim=2)
        #cell = torch.cat([cell_raw, cell_mean], dim=2)
        
        decoder_input_head = x[:,-self.augmented_input_size:,-1]
        decoder_input_tail = (x[:,[-1],-1] - x[:,[-2],-1])/self.control_dt
        outputs = []
        for i in range(self.target_size):
            decoder_output, (hidden, cell) = self.lstm_decoder(torch.cat([decoder_input_head,decoder_input_tail],dim=1).unsqueeze(1), (hidden, cell))
            decoder_output = self.post_decoder(decoder_output)
            decoder_output = decoder_output * post_decoder_scales
            output = self.finalize(decoder_output)[:,:,0] * self.input_rate_limit
        
            outputs.append(output)
            if true_prediction is not None:
                decoder_input_head = decoder_input_head @ self.input_transition_matrix + true_prediction[:,i,:] @ self.add_new_input * self.control_dt
                decoder_input_tail = true_prediction[:,i,:]
            else:
                decoder_input_head = decoder_input_head @ self.input_transition_matrix + output @ self.add_new_input * self.control_dt
                decoder_input_tail = output
        outputs = torch.cat(outputs, dim=1).unsqueeze(2)
        return outputs
    
input_step = max(parameters.controller_acc_input_history_len, parameters.controller_steer_input_history_len)
output_step = max(parameters.acc_input_schedule_prediction_len, parameters.steer_input_schedule_prediction_len)
control_dt = 0.033
acc_cmd_smoothing_sigma = 10.0
steer_cmd_smoothing_sigma = 10.0

def data_smoothing(data: np.ndarray, sigma: float) -> np.ndarray:
    """Apply a Gaussian filter to the data."""
    data_ = gaussian_filter(data, sigma)
    return data_

class AddDataFromCsv:
    def __init__(self):
        self.X_acc_train_list = []
        self.Y_acc_train_list = []
        self.X_steer_train_list = []
        self.Y_steer_train_list = []
        self.indices_train_list = []
        self.X_acc_val_list = []
        self.Y_acc_val_list = []
        self.X_steer_val_list = []
        self.Y_steer_val_list = []
        self.indices_val_list = []
        self.dataset_num = 0
    def clear_data(self):
        self.X_acc_train_list = []
        self.Y_acc_train_list = []
        self.X_steer_train_list = []
        self.Y_steer_train_list = []
        self.indices_train_list = []
        self.X_acc_val_list = []
        self.Y_acc_val_list = []
        self.X_steer_val_list = []
        self.Y_steer_val_list = []
        self.indices_val_list = []
        self.dataset_num = 0
    def add_data_from_csv(self, dir_name, add_mode="as_train",control_cmd_mode=None,dataset_idx=0,reverse_steer=False):
        localization_kinematic_state = np.loadtxt(
            dir_name + "/localization_kinematic_state.csv", delimiter=",", usecols=[0, 1, 4, 5, 7, 8, 9, 10, 47]
        )
        vel = localization_kinematic_state[:, 8]
        if control_cmd_mode == "compensated_control_cmd":
            control_cmd = np.loadtxt(
                dir_name + "/vehicle_raw_vehicle_cmd_converter_debug_compensated_control_cmd.csv", delimiter=",", usecols=[0, 1, 8, 16]
            )
        elif control_cmd_mode == "control_command":
            control_cmd = np.loadtxt(
                dir_name + "/control_command_control_cmd.csv", delimiter=",", usecols=[0, 1, 8, 16]
            )
        elif control_cmd_mode == "control_trajectory_follower":
            control_cmd = np.loadtxt(
                dir_name + "/control_trajectory_follower_control_cmd.csv", delimiter=",", usecols=[0, 1, 8, 16]
            )
        elif control_cmd_mode == "external_selected":
            control_cmd = np.loadtxt(
                dir_name + "/external_selected_control_cmd.csv", delimiter=",", usecols=[0, 1, 8, 16]
            )
        elif control_cmd_mode is None:
            if os.path.exists(dir_name + '/control_command_control_cmd.csv'):
                control_cmd = np.loadtxt(
                    dir_name + "/control_command_control_cmd.csv", delimiter=",", usecols=[0, 1, 8, 16]
                )
            elif os.path.exists(dir_name + '/control_trajectory_follower_control_cmd.csv'):
                control_cmd = np.loadtxt(
                    dir_name + "/control_trajectory_follower_control_cmd.csv", delimiter=",", usecols=[0, 1, 8, 16]
                )
            elif os.path.exists(dir_name + '/external_selected_control_cmd.csv'):
                control_cmd = np.loadtxt(
                    dir_name + "/external_selected_control_cmd.csv", delimiter=",", usecols=[0, 1, 8, 16]
                )
            elif os.path.exists(dir_name + "/vehicle_raw_vehicle_cmd_converter_debug_compensated_control_cmd.csv"):
                control_cmd = np.loadtxt(
                    dir_name + "/vehicle_raw_vehicle_cmd_converter_debug_compensated_control_cmd.csv", delimiter=",", usecols=[0, 1, 8, 16]
                )
            else:
                print("control command csv is not found")
                return
        else:
            print("control_cmd_mode is invalid")
            return
        acc_cmd = control_cmd[:, 3]
        steer_cmd = control_cmd[:, 2]
        system_operation_mode_state = np.loadtxt(
            dir_name + "/system_operation_mode_state.csv", delimiter=",", usecols=[0, 1, 2]
        )
        if system_operation_mode_state.ndim == 1:
            system_operation_mode_state = system_operation_mode_state.reshape(1, -1)
        with open(dir_name + "/system_operation_mode_state.csv") as f:
            reader = csv.reader(f, delimiter=",")
            autoware_control_enabled_str = np.array([row[3] for row in reader])

        control_enabled = np.zeros(system_operation_mode_state.shape[0])
        for i in range(system_operation_mode_state.shape[0]):
            if system_operation_mode_state[i, 2] > 1.5 and autoware_control_enabled_str[i] == "True":
                control_enabled[i] = 1.0
        for i in range(system_operation_mode_state.shape[0] - 1):
            if control_enabled[i] < 0.5 and control_enabled[i + 1] > 0.5:
                operation_start_time = system_operation_mode_state[i + 1, 0] + 1e-9 * system_operation_mode_state[i + 1, 1]
            elif control_enabled[i] > 0.5 and control_enabled[i + 1] < 0.5:
                operation_end_time = system_operation_mode_state[i + 1, 0] + 1e-9 * system_operation_mode_state[i + 1, 1]
                break
            operation_end_time = localization_kinematic_state[-1, 0] + 1e-9 * localization_kinematic_state[-1, 1]
        if system_operation_mode_state.shape[0] == 1:
            operation_end_time = localization_kinematic_state[-1, 0] + 1e-9 * localization_kinematic_state[-1, 1]
        if control_enabled[0] > 0.5:
            operation_start_time = system_operation_mode_state[0, 0] + 1e-9 * system_operation_mode_state[0, 1]
        print("operation_start_time", operation_start_time)
        print("operation_end_time", operation_end_time)

        min_time_stamp = max(
            [
                operation_start_time,
                localization_kinematic_state[0, 0] + 1e-9 * localization_kinematic_state[0, 1],
                control_cmd[0, 0] + 1e-9 * control_cmd[0, 1],
            ]
        )
        max_time_stamp = min(
            [
                operation_end_time,
                localization_kinematic_state[-1, 0] + 1e-9 * localization_kinematic_state[-1, 1],
                control_cmd[-1, 0] + 1e-9 * control_cmd[-1, 1],
            ]
        )
        data_num = int((max_time_stamp - min_time_stamp)/control_dt)
        data_time_stamps = min_time_stamp + control_dt * np.arange(data_num)
        vel_interp = scipy.interpolate.interp1d(localization_kinematic_state[:, 0] + 1e-9 * localization_kinematic_state[:, 1], vel)(data_time_stamps)
        acc_cmd_interp = scipy.interpolate.interp1d(control_cmd[:, 0] + 1e-9 * control_cmd[:, 1], acc_cmd)(data_time_stamps)
        steer_cmd_interp = scipy.interpolate.interp1d(control_cmd[:, 0] + 1e-9 * control_cmd[:, 1], steer_cmd)(data_time_stamps)
        acc_cmd_smoothed = data_smoothing(acc_cmd_interp, acc_cmd_smoothing_sigma)
        steer_cmd_smoothed = data_smoothing(steer_cmd_interp, steer_cmd_smoothing_sigma)
        X_acc = []
        X_steer = []
        Y_acc = []
        Y_steer = []
        indices = []
        for i in range(data_num - input_step - output_step):
            if vel_interp[i + input_step] < 0.1:
                continue
            X_acc.append(
                np.stack(
                    [
                        vel_interp[i:i + input_step],
                        acc_cmd_interp[i:i + input_step],
                    ]
                ).T
            )
            if reverse_steer:
                X_steer.append(
                    np.stack(
                        [
                            vel_interp[i:i + input_step],
                            -steer_cmd_interp[i:i + input_step],
                        ]
                    ).T
                )
                Y_steer.append(
                    - np.array(steer_cmd_smoothed[i + input_step:i + input_step + output_step] - steer_cmd_smoothed[i + input_step - 1:i + input_step + output_step - 1]).reshape(-1, 1)/control_dt
                )
            else:
                X_steer.append(
                    np.stack(
                        [
                            vel_interp[i:i + input_step],
                            steer_cmd_interp[i:i + input_step],
                        ]
                    ).T
                )
                Y_steer.append(
                    np.array(steer_cmd_smoothed[i + input_step:i + input_step + output_step] - steer_cmd_smoothed[i + input_step - 1:i + input_step + output_step - 1]).reshape(-1, 1)/control_dt
                )

            Y_acc.append(
                np.array(acc_cmd_smoothed[i + input_step:i + input_step + output_step] - acc_cmd_smoothed[i + input_step - 1:i + input_step + output_step - 1]).reshape(-1, 1)/control_dt
            )



            indices.append(dataset_idx)
        if dataset_idx not in self.indices_train_list and dataset_idx not in self.indices_val_list:
            self.dataset_num += 1
        if add_mode == "as_train":
            self.X_acc_train_list += X_acc
            self.Y_acc_train_list += Y_acc
            self.X_steer_train_list += X_steer
            self.Y_steer_train_list += Y_steer
            self.indices_train_list += indices
        elif add_mode == "as_val":
            self.X_acc_val_list += X_acc
            self.Y_acc_val_list += Y_acc
            self.X_steer_val_list += X_steer
            self.Y_steer_val_list += Y_steer
            self.indices_val_list += indices
        else:
            print("add_mode is invalid")
            return
        
def generate_random_vector(batch_size, seq_len, state_dim, dt, device, vel_scaling):
    random_vector = torch.randn(batch_size, seq_len, state_dim, device=device)
    random_vector[:,1:,:] = random_vector[:,1:,:] * dt
    random_vector[:,:,0] = random_vector[:,:,0] / vel_scaling
    random_vector_integrated = torch.cumsum(random_vector, dim=1)
    return random_vector_integrated

def get_loss(criterion,model,X,Y,indices, stage_error_weight=1e-4, prediction_error_weight=100.0,max_small_input_weight= 50.0,small_input_threshold=0.01,second_order_weight=1e-2, integral_points = [0.2,0.5,1.0], integral_weights = [1e-3,1e-2,1.0] ,tanh_gain = 2.0, tanh_weight = 0.1, use_true_prediction=False, alpha_jacobian=None, eps=1e-5):
    device = X.device
    if alpha_jacobian is not None:
        random_vector = generate_random_vector(X.size(0),X.size(1),X.size(2),control_dt,device,model.vel_scaling) * eps
    if use_true_prediction:
        Y_pred = model(X,indices,Y)
        if alpha_jacobian is not None:
            Y_perturbed = model(X+random_vector,indices,Y)
    else:
        Y_pred = model(X,indices)
        if alpha_jacobian is not None:
            Y_perturbed = model(X+random_vector,indices)
    loss = stage_error_weight * criterion(Y_pred,Y)
    if alpha_jacobian is not None:
        loss = loss + alpha_jacobian * torch.mean(torch.abs((Y_perturbed - Y_pred) / eps))
    input_future = torch.mean(torch.abs(X[:,-1,1] + torch.cumsum(Y, dim=1) * control_dt), dim=1)
    small_input_weight = torch.where(input_future > small_input_threshold, 
                                    torch.tensor(1.0, device=X.device), 
                                    1.0/(input_future + 1.0/max_small_input_weight))

    for i in range(len(integral_points)):
        loss = loss + integral_weights[i] * tanh_weight * torch.mean(torch.abs(torch.tanh(tanh_gain * (Y_pred[:,:int(model.target_size*integral_points[i]),0].sum(dim=1) - Y[:,:int(model.target_size*integral_points[i]),0].sum(dim=1)))))
        loss = loss + integral_weights[i] * prediction_error_weight * torch.mean(small_input_weight * torch.abs(Y_pred[:,:int(model.target_size*integral_points[i])].sum(dim=1)*control_dt - Y[:,:int(model.target_size*integral_points[i])].sum(dim=1)*control_dt))

    #loss = loss + second_order_weight * criterion((Y_pred[:,1:] - Y_pred[:,:-1])/control_dt,(Y[:,1:] - Y[:,:-1])/control_dt)
    loss = loss + second_order_weight * torch.mean(torch.abs((Y_pred[:,1:] - Y_pred[:,:-1])/control_dt)) #criterion((Y_pred[:,1:] - Y_pred[:,:-1])/control_dt,(Y[:,1:] - Y[:,:-1])/control_dt)
    loss = loss + second_order_weight * torch.mean(torch.abs(Y_pred[:,-1]/control_dt)) / model.target_size
    return loss
def validate_in_batches(criterion,model,X_val,Y_val,indices,batch_size=1000, tanh_gain = 2.0, tanh_weight = 0.1,max_small_input_weight= 50.0,small_input_threshold=0.01, alpha_jacobian=None):
    model.eval()
    val_loss = 0.0
    num_batches = (X_val.size(0) + batch_size - 1) // batch_size
    with torch.no_grad():
        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size
            X_val_batch = X_val[start:end]
            Y_val_batch = Y_val[start:end]
            indices_batch = indices[start:end]
            loss = get_loss(criterion,model,X_val_batch,Y_val_batch, indices_batch, tanh_gain=tanh_gain, tanh_weight=tanh_weight, max_small_input_weight=max_small_input_weight, small_input_threshold=small_input_threshold, alpha_jacobian=alpha_jacobian).item()
            val_loss += loss *(end - start)
    val_loss /= X_val.size(0)
    return val_loss
class TrainInputsSchedulePredictorNN(AddDataFromCsv):
    def __init__(self, max_iter=10000, tol=1e-5, alpha_1=0.1**7, alpha_2=0.1**7, tanh_gain_acc=2.0, tanh_weight_acc=0.1,  tanh_gain_steer=10.0, tanh_weight_steer=0.01,alpha_jacobian=None):
        super().__init__()
        self.max_iter = max_iter
        self.tol = tol
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.alpha_jacobian = alpha_jacobian
        self.tanh_gain_acc = tanh_gain_acc
        self.tanh_gain_steer = tanh_gain_steer
        self.tanh_weight_acc = tanh_weight_acc
        self.tanh_weight_steer = tanh_weight_steer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_acc = None
        self.model_steer = None
    def train_model(
            self,
            model,
            X_train,
            Y_train,
            indices_train,
            X_val,
            Y_val,
            indices_val,
            batch_sizes,
            learning_rates,
            patience,
            tanh_gain,
            tanh_weight,
            max_small_input_weight=50.0,
            small_input_threshold=0.01,
    ):
        print("sample_size:", X_train.shape[0] + X_val.shape[0])
        print("patience:", patience)
        # Define the loss function.
        criterion = nn.L1Loss()
        # Define the optimizer.
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rates[0])
        # Define the initial loss.
        initial_loss = validate_in_batches(criterion,model,X_val,Y_val,indices_val, tanh_gain=tanh_gain, tanh_weight=tanh_weight, max_small_input_weight=max_small_input_weight, small_input_threshold=small_input_threshold) #,alpha_jacobian=self.alpha_jacobian)
        print("initial_loss:", initial_loss)
        batch_size = batch_sizes[0]

        print("batch_size:", batch_size)
        # Define the early stopping object.
        early_stopping = EarlyStopping(initial_loss, tol=self.tol, patience=patience)
        # Data loader for training.
        train_dataset = torch.utils.data.TensorDataset(X_train, Y_train, indices_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        # learning_rate index
        learning_rate_index = 0
        print("learning rate:", learning_rates[learning_rate_index])
        # Training loop.
        for i in range(self.max_iter):
            model.train()
            for X_batch, Y_batch, indices_batch in train_loader:
                optimizer.zero_grad()
                if learning_rates[learning_rate_index] > 5e-3:
                    use_true_prediction = True
                else:
                    use_true_prediction = False
                loss = get_loss(criterion,model,X_batch,Y_batch,indices_batch,use_true_prediction=use_true_prediction, tanh_gain=tanh_gain, tanh_weight=tanh_weight, max_small_input_weight=max_small_input_weight, small_input_threshold=small_input_threshold,alpha_jacobian=self.alpha_jacobian)
                for w in model.named_parameters():
                    if w[0].find("adaptive_scales") == -1:
                        loss += self.alpha_1 * torch.sum(w[1]**2) + self.alpha_2 * torch.sum(torch.abs(w[1]))
                for j in range(self.dataset_num):
                    index_rates = (indices_batch == j).sum().item()/indices_batch.size(0)
                    loss += self.alpha_1 * index_rates * torch.mean(torch.abs(model.post_decoder_adaptive_scales[j] - 1.0))
    
                loss.backward()
                #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                optimizer.step()
            # Validation loss.
            model.eval()
            val_loss = validate_in_batches(criterion,model,X_val,Y_val, indices_val, tanh_gain=tanh_gain, tanh_weight=tanh_weight, max_small_input_weight=max_small_input_weight, small_input_threshold=small_input_threshold) #,alpha_jacobian=self.alpha_jacobian)
            if i % 10 == 0:
                print(val_loss, i)
            if early_stopping(val_loss):
                learning_rate_index += 1
                batch_size = batch_sizes[min(learning_rate_index, len(batch_sizes) - 1)]
                if learning_rate_index >= len(learning_rates):
                    break
                else:
                    print("update learning rate to ", learning_rates[learning_rate_index])
                    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rates[learning_rate_index])
                    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                    early_stopping.reset()
    def relearn_model(
            self,
            model,
            X_train,
            Y_train,
            indices_train,
            X_val,
            Y_val,
            indices_val,
            batch_sizes,
            learning_rates,
            patience,
            tanh_gain,
            tanh_weight,
            max_small_input_weight=50.0,
            small_input_threshold=0.01,
            randomize=0.001
    ):  
        model.lstm_decoder.flatten_parameters()
        model.lstm_encoder.flatten_parameters()
        # Define the loss function.
        criterion = nn.L1Loss()
        # Define the optimizer.
        # Define the initial loss.
        original_val_loss = validate_in_batches(criterion,model,X_val,Y_val,indices_val, tanh_gain=tanh_gain, tanh_weight=tanh_weight, max_small_input_weight=max_small_input_weight, small_input_threshold=small_input_threshold) #,alpha_jacobian=self.alpha_jacobian)
        relearned_model = copy.deepcopy(model)
        relearned_model.lstm_decoder.flatten_parameters()
        relearned_model.lstm_encoder.flatten_parameters()
        with torch.no_grad():
            for w in relearned_model.parameters():
                w += randomize * torch.randn_like(w)
        self.train_model(
            relearned_model,
            X_train,
            Y_train,
            indices_train,
            X_val,
            Y_val,
            indices_val,
            batch_sizes,
            learning_rates,
            patience,
            tanh_gain,
            tanh_weight,
            max_small_input_weight=max_small_input_weight,
            small_input_threshold=small_input_threshold
        )
        relearned_val_loss = validate_in_batches(criterion,relearned_model,X_val,Y_val,indices_val, tanh_gain=tanh_gain, tanh_weight=tanh_weight, max_small_input_weight=max_small_input_weight, small_input_threshold=small_input_threshold) #,alpha_jacobian=self.alpha_jacobian)
        print("original_val_loss:", original_val_loss)
        print("relearned_val_loss:", relearned_val_loss)
        if relearned_val_loss < original_val_loss:
            return relearned_model
        else:
            return model
                    
    def get_trained_model(self,learning_rates=[1e-3,1e-4,1e-5,1e-6],patience=10,batch_sizes=[1000], jerk_limit=1.0, steer_rate_limit=1.0, augmented_input_size=15,max_small_acc_weight=10.0, small_acc_threshold=0.01 ,
                          max_small_steer_weight=30.0, small_steer_threshold=0.005, cmd_mode="both"):
        if cmd_mode == "both" or cmd_mode == "acc":
            self.model_acc = InputsSchedulePredictorNN(dataset_num=self.dataset_num,target_size=parameters.acc_input_schedule_prediction_len, input_rate_limit=jerk_limit, augmented_input_size=augmented_input_size).to(self.device)
            self.train_model(
                self.model_acc,
                torch.tensor(np.array(self.X_acc_train_list)[:,output_step - parameters.controller_acc_input_history_len:], dtype=torch.float32,device=self.device),
                torch.tensor(np.array(self.Y_acc_train_list)[:,:parameters.acc_input_schedule_prediction_len], dtype=torch.float32,device=self.device),
                torch.tensor(np.array(self.indices_train_list), dtype=torch.long,device=self.device),
                torch.tensor(np.array(self.X_acc_val_list)[:,output_step - parameters.controller_acc_input_history_len:], dtype=torch.float32,device=self.device),
                torch.tensor(np.array(self.Y_acc_val_list)[:,:parameters.acc_input_schedule_prediction_len], dtype=torch.float32,device=self.device),
                torch.tensor(np.array(self.indices_val_list), dtype=torch.long,device=self.device),
                batch_sizes,
                learning_rates,
                patience,
                self.tanh_gain_acc,
                self.tanh_weight_acc,
                max_small_input_weight=max_small_acc_weight,
                small_input_threshold=small_acc_threshold,
            )
        if cmd_mode == "both" or cmd_mode == "steer":
            self.model_steer = InputsSchedulePredictorNN(dataset_num=self.dataset_num,target_size=parameters.steer_input_schedule_prediction_len, input_rate_limit=steer_rate_limit,augmented_input_size=augmented_input_size).to(self.device)
            self.train_model(
                self.model_steer,
                torch.tensor(np.array(self.X_steer_train_list)[:,output_step - parameters.controller_steer_input_history_len:], dtype=torch.float32,device=self.device),
                torch.tensor(np.array(self.Y_steer_train_list)[:,:parameters.steer_input_schedule_prediction_len], dtype=torch.float32,device=self.device),
                torch.tensor(np.array(self.indices_train_list), dtype=torch.long,device=self.device),
                torch.tensor(np.array(self.X_steer_val_list)[:,output_step - parameters.controller_steer_input_history_len:], dtype=torch.float32,device=self.device),
                torch.tensor(np.array(self.Y_steer_val_list)[:,:parameters.steer_input_schedule_prediction_len], dtype=torch.float32,device=self.device),
                torch.tensor(np.array(self.indices_val_list), dtype=torch.long,device=self.device),
                batch_sizes,
                learning_rates,
                patience,
                self.tanh_gain_steer,
                self.tanh_weight_steer,
                max_small_input_weight=max_small_steer_weight,
                small_input_threshold=small_steer_threshold,
            )
    def relearn_acc(self,learning_rates=[1e-3,1e-4,1e-5,1e-6],patience=10,batch_sizes=[1000], jerk_limit=1.0, max_small_acc_weight=10.0, small_acc_threshold=0.01, randomize=0.001):
        self.model_acc.to(self.device)
        self.model_acc = self.relearn_model(
            self.model_acc,
            torch.tensor(np.array(self.X_acc_train_list), dtype=torch.float32,device=self.device),
            torch.tensor(np.array(self.Y_acc_train_list), dtype=torch.float32,device=self.device),
            torch.tensor(np.array(self.indices_train_list), dtype=torch.long,device=self.device),
            torch.tensor(np.array(self.X_acc_val_list), dtype=torch.float32,device=self.device),
            torch.tensor(np.array(self.Y_acc_val_list), dtype=torch.float32,device=self.device),
            torch.tensor(np.array(self.indices_val_list), dtype=torch.long,device=self.device),
            batch_sizes,
            learning_rates,
            patience,
            self.tanh_gain_acc,
            self.tanh_weight_acc,
            max_small_input_weight=max_small_acc_weight,
            small_input_threshold=small_acc_threshold,
            randomize=randomize
        )
    def relearn_steer(self,learning_rates=[1e-3,1e-4,1e-5,1e-6],patience=10,batch_sizes=[1000], steer_rate_limit=1.0, max_small_steer_weight=30.0, small_steer_threshold=0.005, randomize=0.001):
        self.model_steer.to(self.device)
        self.model_steer = self.relearn_model(
            self.model_steer,
            torch.tensor(np.array(self.X_steer_train_list), dtype=torch.float32,device=self.device),
            torch.tensor(np.array(self.Y_steer_train_list), dtype=torch.float32,device=self.device),
            torch.tensor(np.array(self.indices_train_list), dtype=torch.long,device=self.device),
            torch.tensor(np.array(self.X_steer_val_list), dtype=torch.float32,device=self.device),
            torch.tensor(np.array(self.Y_steer_val_list), dtype=torch.float32,device=self.device),
            torch.tensor(np.array(self.indices_val_list), dtype=torch.long,device=self.device),
            batch_sizes,
            learning_rates,
            patience,
            self.tanh_gain_steer,
            self.tanh_weight_steer,
            max_small_input_weight=max_small_steer_weight,
            small_input_threshold=small_steer_threshold,
            randomize=randomize
        )
    def save_model(self,path="inputs_schedule_predictor_model",cmd_mode="both"):
        if not os.path.exists(path):
            os.makedirs(path)
        if cmd_mode == "both" or cmd_mode == "acc":
            self.model_acc.to("cpu")
            torch.save(self.model_acc, path + "/acc_schedule_predictor.pth")
            convert_model_to_csv.convert_inputs_schedule_model_to_csv(self.model_acc, path + "/acc_schedule_predictor")
        if cmd_mode == "both" or cmd_mode == "steer":
            self.model_steer.to("cpu")
            torch.save(self.model_steer, path + "/steer_schedule_predictor.pth")
            convert_model_to_csv.convert_inputs_schedule_model_to_csv(self.model_steer, path + "/steer_schedule_predictor")
