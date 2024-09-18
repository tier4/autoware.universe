import numpy as np
import torch
import torch.nn as nn
import os
import csv
import scipy.interpolate
from autoware_vehicle_adaptor.training.early_stopping import EarlyStopping
from autoware_vehicle_adaptor.training import convert_model_to_csv
from scipy.ndimage import gaussian_filter

class InputsSchedulePredictorNN(nn.Module):
    def __init__(self, dataset_num,pre_encoder_hidden_size=(32,16),lstm_hidden_size=64,#16,
                 post_decoder_hidden_size=(64,16),control_dt=0.033,target_size=35, jerk_lim=1.0,steer_rate_lim=1.0,
                 vel_scaling=0.1, vel_bias=0.0):#,vel_bias=5.0):
        super(InputsSchedulePredictorNN, self).__init__()
        self.dataset_num = dataset_num
        self.control_dt = control_dt
        self.target_size = target_size
        self.input_size = 5
        self.output_size = 2
        self.vel_scaling = vel_scaling
        self.vel_bias = vel_bias
        self.pre_encoder = nn.Sequential(
            nn.Linear(self.input_size, pre_encoder_hidden_size[0]),
            nn.ReLU(),
            nn.Linear(pre_encoder_hidden_size[0], pre_encoder_hidden_size[1]),
            nn.ReLU()
        )
        self.lstm_encoder = nn.LSTM(pre_encoder_hidden_size[1], lstm_hidden_size, batch_first=True)
        self.lstm_decoder = nn.LSTM(2 * self.output_size, lstm_hidden_size, batch_first=True)
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
        self.jerk_lim = jerk_lim
        self.steer_rate_lim = steer_rate_lim
    def forward(self, x, dataset_idx, true_prediction=None):
        device = x.device
        limit_scaling = torch.tensor([self.jerk_lim, self.steer_rate_lim], device=device)
        x_vel_scaled = x.clone()
        x_vel_scaled[:,:,0] = (x_vel_scaled[:,:,0] - self.vel_bias ) * self.vel_scaling
        x_vel_scaled = self.pre_encoder(x_vel_scaled)
        post_decoder_scales_tensor = torch.stack(list(self.post_decoder_adaptive_scales)).to(device)
        post_decoder_scales = post_decoder_scales_tensor[dataset_idx]
        post_decoder_scales = post_decoder_scales.unsqueeze(1)
        _, (hidden, cell) = self.lstm_encoder(x_vel_scaled)
        #decoder_input = (x[:,[-1],-2:] - x[:,[-2],-2:])/self.control_dt
        #decoder_input = torch.zeros(x.size(0),1,self.output_size,device=device)
        #decoder_input = x[:,[-1],-2:]
        # decoder_input = torch.cat([x[:,[-1],-2:], (x[:,[-1],-2:] - x[:,[-2],-2:])/self.control_dt ],dim=2)
        decoder_input_head = x[:,[-1],-2:]
        decoder_input_tail = (x[:,[-1],-2:] - x[:,[-2],-2:])/self.control_dt
        outputs = []
        for i in range(self.target_size):
            decoder_output, (hidden, cell) = self.lstm_decoder(torch.cat([decoder_input_head,decoder_input_tail],dim=2), (hidden, cell))
            #(decoder_input, (hidden,cell))
            decoder_output = self.post_decoder(decoder_output)
            decoder_output = decoder_output * post_decoder_scales
            output = self.finalize(decoder_output) * limit_scaling# * torch.tensor([self.jerk_lim, self.steer_rate_lim], device=device)
        
            outputs.append(output)
            if true_prediction is not None:
                # decoder_input = true_prediction[:,[i],:]
                # decoder_input = decoder_input + true_prediction[:,[i],:] * self.control_dt
                decoder_input_head = decoder_input_head + true_prediction[:,[i],:] * self.control_dt
                decoder_input_tail = true_prediction[:,[i],:]

            else:
                # decoder_input = output
                # decoder_input = decoder_input + output * self.control_dt
                decoder_input_head = decoder_input_head + output * self.control_dt
                decoder_input_tail = output

        outputs = torch.cat(outputs, dim=1)
        return outputs
    
input_step = 150
output_step = 35
control_dt = 0.033
acc_cmd_smoothing_sigma = 10.0
steer_cmd_smoothing_sigma = 10.0

def data_smoothing(data: np.ndarray, sigma: float) -> np.ndarray:
    """Apply a Gaussian filter to the data."""
    data_ = gaussian_filter(data, sigma)
    return data_

class AddDataFromCsv:
    def __init__(self):
        self.X_train_list = []
        self.Y_train_list = []
        self.indices_train_list = []
        self.X_val_list = []
        self.Y_val_list = []
        self.indices_val_list = []
        self.dataset_num = 0
    def clear_data(self):
        self.X_train_list = []
        self.Y_train_list = []
        self.X_val_list = []
        self.Y_val_list = []
        self.indices_train_list = []
        self.indices_val_list = []
        self.dataset_num = 0
    def add_data_from_csv(self, dir_name, add_mode="as_train",control_cmd_mode=None,dataset_idx=0):
        localization_kinematic_state = np.loadtxt(
            dir_name + "/localization_kinematic_state.csv", delimiter=",", usecols=[0, 1, 4, 5, 7, 8, 9, 10, 47]
        )
        vel = localization_kinematic_state[:, 8]
        localization_acceleration = np.loadtxt(dir_name + "/localization_acceleration.csv", delimiter=",", usecols=[0, 1, 3])
        acc = localization_acceleration[:, 2]
        vehicle_status_steering_status = np.loadtxt(
            dir_name + "/vehicle_status_steering_status.csv", delimiter=",", usecols=[0, 1, 2]
        )
        steer = vehicle_status_steering_status[:, 2]
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
                localization_acceleration[0, 0] + 1e-9 * localization_acceleration[0, 1],
                vehicle_status_steering_status[0, 0] + 1e-9 * vehicle_status_steering_status[0, 1],
                control_cmd[0, 0] + 1e-9 * control_cmd[0, 1],
            ]
        )
        max_time_stamp = min(
            [
                operation_end_time,
                localization_kinematic_state[-1, 0] + 1e-9 * localization_kinematic_state[-1, 1],
                localization_acceleration[-1, 0] + 1e-9 * localization_acceleration[-1, 1],
                vehicle_status_steering_status[-1, 0] + 1e-9 * vehicle_status_steering_status[-1, 1],
                control_cmd[-1, 0] + 1e-9 * control_cmd[-1, 1],
            ]
        )
        data_num = int((max_time_stamp - min_time_stamp)/control_dt)
        data_time_stamps = min_time_stamp + control_dt * np.arange(data_num)
        vel_interp = scipy.interpolate.interp1d(localization_kinematic_state[:, 0] + 1e-9 * localization_kinematic_state[:, 1], vel)(data_time_stamps)
        acc_interp = scipy.interpolate.interp1d(localization_acceleration[:, 0] + 1e-9 * localization_acceleration[:, 1], acc)(data_time_stamps)
        steer_interp = scipy.interpolate.interp1d(vehicle_status_steering_status[:, 0] + 1e-9 * vehicle_status_steering_status[:, 1], steer)(data_time_stamps)
        acc_cmd_interp = scipy.interpolate.interp1d(control_cmd[:, 0] + 1e-9 * control_cmd[:, 1], acc_cmd)(data_time_stamps)
        steer_cmd_interp = scipy.interpolate.interp1d(control_cmd[:, 0] + 1e-9 * control_cmd[:, 1], steer_cmd)(data_time_stamps)
        acc_cmd_smoothed = data_smoothing(acc_cmd_interp, acc_cmd_smoothing_sigma)
        steer_cmd_smoothed = data_smoothing(steer_cmd_interp, steer_cmd_smoothing_sigma)
        X = []
        Y = []
        indices = []
        for i in range(data_num - input_step - output_step):
            if vel_interp[i + input_step] < 0.1:
                continue
            X.append(
                np.stack(
                    [
                        vel_interp[i:i + input_step],
                        acc_interp[i:i + input_step],
                        steer_interp[i:i + input_step],
                        acc_cmd_interp[i:i + input_step],
                        steer_cmd_interp[i:i + input_step],
                    ]
                ).T
            )

            Y.append(
                np.stack(
                    [
                        (acc_cmd_smoothed[i + input_step:i + input_step + output_step] - acc_cmd_smoothed[i + input_step - 1:i + input_step + output_step - 1])/control_dt,
                        (steer_cmd_smoothed[i + input_step:i + input_step + output_step] - steer_cmd_smoothed[i + input_step - 1:i + input_step + output_step - 1])/control_dt,
                    ]
                ).T
            )
            indices.append(dataset_idx)
        if dataset_idx not in self.indices_train_list and dataset_idx not in self.indices_val_list:
            self.dataset_num += 1
        if add_mode == "as_train":
            self.X_train_list += X
            self.Y_train_list += Y
            self.indices_train_list += indices
        elif add_mode == "as_val":
            self.X_val_list += X
            self.Y_val_list += Y
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

def get_loss(criterion,model,X,Y,indices, stage_error_weight=1e-4, prediction_error_weight=100.0,second_order_weight=1e-2, integral_points = [0.2,0.5,1.0], integral_weights = [1e-3,1e-2,1.0] ,tanh_gain_acc = 2.0, tanh_gain_steer = 20.0, tanh_weight_acc = 0.1, tanh_weight_steer = 10.0, use_true_prediction=False, alpha_jacobian=None, eps=1e-5):
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
        loss = loss + alpha_jacobian * torch.mean(torch.abs((Y_perturbed - Y) / eps))
    
    #loss = loss + tanh_weight_acc * torch.mean(torch.abs(torch.tanh(tanh_gain_acc * (Y_pred[:,:output_step//2,0].sum(dim=1) - Y[:,:output_step//2,0].sum(dim=1)))))
    #loss = loss + tanh_weight_steer * torch .mean(torch.abs(torch.tanh(tanh_gain_steer * (Y_pred[:,:output_step//2,1].sum(dim=1) - Y[:,:output_step//2,1].sum(dim=1))*control_dt)))
    #loss = loss + prediction_error_weight * criterion(Y_pred[:,:output_step//2].sum(dim=1)*control_dt,Y[:,:output_step//2].sum(dim=1)*control_dt)
    #loss = loss + tanh_weight_acc * torch.mean(torch.abs(torch.tanh(tanh_gain_acc * (Y_pred[:,:,0].sum(dim=1) - Y[:,:,0].sum(dim=1)))))
    #loss = loss + tanh_weight_steer * torch.mean(torch.abs(torch.tanh(tanh_gain_steer * (Y_pred[:,:,1].sum(dim=1) - Y[:,:,1].sum(dim=1))*control_dt)))
    #loss = loss + prediction_error_weight * criterion(Y_pred.sum(dim=1)*control_dt,Y.sum(dim=1)*control_dt)
    for i in range(len(integral_points)):
        loss = loss + integral_weights[i] * tanh_weight_acc * torch.mean(torch.abs(torch.tanh(tanh_gain_acc * (Y_pred[:,:int(output_step*integral_points[i]),0].sum(dim=1) - Y[:,:int(output_step*integral_points[i]),0].sum(dim=1)))))
        loss = loss + integral_weights[i] * tanh_weight_steer * torch.mean(torch.abs(torch.tanh(tanh_gain_steer * (Y_pred[:,:int(output_step*integral_points[i]),1].sum(dim=1) - Y[:,:int(output_step*integral_points[i]),1].sum(dim=1))*control_dt)))
        loss = loss + integral_weights[i] * prediction_error_weight * criterion(Y_pred[:,:int(output_step*integral_points[i])].sum(dim=1)*control_dt,Y[:,:int(output_step*integral_points[i])].sum(dim=1)*control_dt)

    #loss = loss + second_order_weight * criterion((Y_pred[:,1:] - Y_pred[:,:-1])/control_dt,(Y[:,1:] - Y[:,:-1])/control_dt)
    loss = loss + second_order_weight * torch.mean(torch.abs((Y_pred[:,1:] - Y_pred[:,:-1])/control_dt)) #criterion((Y_pred[:,1:] - Y_pred[:,:-1])/control_dt,(Y[:,1:] - Y[:,:-1])/control_dt)
    return loss
def validate_in_batches(criterion,model,X_val,Y_val,indices,batch_size=1000, alpha_jacobian=None):
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
            loss = get_loss(criterion,model,X_val_batch,Y_val_batch, indices_batch, alpha_jacobian=alpha_jacobian).item()
            val_loss += loss *(end - start)
    val_loss /= X_val.size(0)
    return val_loss
class TrainInputsSchedulePredictorNN(AddDataFromCsv):
    def __init__(self, max_iter=10000, tol=1e-5, alpha_1=0.1**7, alpha_2=0.1**7,alpha_jacobian=None):
        super().__init__()
        self.max_iter = max_iter
        self.tol = tol
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.alpha_jacobian = alpha_jacobian
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
    def train_model(
            self,
            model,
            X_train,
            Y_train,
            indices_train,
            X_val,
            Y_val,
            indices_val,
            batch_size,
            learning_rates,
            patience
    ):
        print("sample_size:", X_train.shape[0] + X_val.shape[0])
        print("patience:", patience)
        # Define the loss function.
        criterion = nn.L1Loss()
        # Define the optimizer.
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rates[0])
        # Define the initial loss.
        initial_loss = validate_in_batches(criterion,model,X_val,Y_val,indices_val)
        print("initial_loss:", initial_loss)
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
                loss = get_loss(criterion,model,X_batch,Y_batch,indices_batch,use_true_prediction=use_true_prediction, alpha_jacobian=self.alpha_jacobian)
                for w in model.named_parameters():
                    if w[0].find("adaptive_scales") == -1:
                        loss += self.alpha_1 * torch.sum(w[1]**2) + self.alpha_2 * torch.sum(torch.abs(w[1]))
                for j in range(self.dataset_num):
                    index_rates = (indices_batch == j).sum().item()/indices_batch.size(0)
                    loss += self.alpha_1 * index_rates * torch.mean(torch.abs(model.post_decoder_adaptive_scales[j] - 1.0))
    
                loss.backward()
                optimizer.step()
            # Validation loss.
            model.eval()
            val_loss = validate_in_batches(criterion,model,X_val,Y_val, indices_val)#,alpha_jacobian=self.alpha_jacobian)
            if i % 10 == 0:
                print(val_loss, i)
            if early_stopping(val_loss):
                learning_rate_index += 1
                if learning_rate_index >= len(learning_rates):
                    break
                else:
                    print("update learning rate to ", learning_rates[learning_rate_index])
                    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rates[learning_rate_index])
                    early_stopping.reset()
    def get_trained_model(self,learning_rates=[1e-3,1e-4,1e-5,1e-6],patience=10,batch_size=1000):
        self.model = InputsSchedulePredictorNN(dataset_num=self.dataset_num,target_size=output_step).to(self.device)
        self.train_model(
            self.model,
            torch.tensor(np.array(self.X_train_list), dtype=torch.float32,device=self.device),
            torch.tensor(np.array(self.Y_train_list), dtype=torch.float32,device=self.device),
            torch.tensor(np.array(self.indices_train_list), dtype=torch.long,device=self.device),
            torch.tensor(np.array(self.X_val_list), dtype=torch.float32,device=self.device),
            torch.tensor(np.array(self.Y_val_list), dtype=torch.float32,device=self.device),
            torch.tensor(np.array(self.indices_val_list), dtype=torch.long,device=self.device),
            batch_size,
            learning_rates,
            patience
        )
    def save_model(self,path="inputs_schedule_predictor.pth"):
        self.model.to("cpu")
        torch.save(self.model.state_dict(), path)
        save_dir = path.replace(".pth", "")
        convert_model_to_csv.convert_inputs_schedule_model_to_csv(self.model, save_dir)
