from autoware_vehicle_adaptor.training import add_data_from_csv
from autoware_vehicle_adaptor.training import error_prediction_NN
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from pathlib import Path
import json
import copy

import yaml
#torch.autograd.set_detect_anomaly(True)
package_path_json = str(Path(__file__).parent.parent) + "/package_path.json"
with open(package_path_json, "r") as file:
    package_path = json.load(file)

trained_model_param_path = (
    package_path["path"] + "/autoware_vehicle_adaptor/param/trained_model_param.yaml"
)
with open(trained_model_param_path, "r") as yml:
    trained_model_param = yaml.safe_load(yml)
prediction_length = int(trained_model_param["trained_model_parameter"]["lstm"]["predict_lstm_len"])
past_length = int(trained_model_param["trained_model_parameter"]["lstm"]["update_lstm_len"])
integration_length = int(trained_model_param["trained_model_parameter"]["loss"]["integration_len"])
integration_weight = float(trained_model_param["trained_model_parameter"]["loss"]["integration_weight"])
add_position_to_prediction = bool(trained_model_param["trained_model_parameter"]["setting"]["add_position_to_prediction"])
add_vel_to_prediction = bool(trained_model_param["trained_model_parameter"]["setting"]["add_vel_to_prediction"])
add_yaw_to_prediction = bool(trained_model_param["trained_model_parameter"]["setting"]["add_yaw_to_prediction"])
integrate_states = bool(trained_model_param["trained_model_parameter"]["setting"]["integrate_states"])
integrate_vel = bool(trained_model_param["trained_model_parameter"]["setting"]["integrate_vel"])
integrate_yaw = bool(trained_model_param["trained_model_parameter"]["setting"]["integrate_yaw"])
if not add_vel_to_prediction:
    integrate_vel = False
if not add_yaw_to_prediction:
    integrate_yaw = False

state_component_predicted =[]
state_component_predicted_index = []
if add_position_to_prediction:
    state_component_predicted.append("x")
    state_component_predicted_index.append(0)
    state_component_predicted.append("y")
    state_component_predicted_index.append(1)
if add_vel_to_prediction:
    state_component_predicted.append("vel")
    state_component_predicted_index.append(2)
if add_yaw_to_prediction:
    state_component_predicted.append("yaw")
    state_component_predicted_index.append(3)
state_component_predicted.append("acc")
state_component_predicted_index.append(4)
state_component_predicted.append("steer")
state_component_predicted_index.append(5)
state_name_to_predicted_index = {}
for i in range(len(state_component_predicted)):
    state_name_to_predicted_index[state_component_predicted[i]] = i

nominal_param_path = (
    package_path["path"] + "/autoware_vehicle_adaptor/param/nominal_param.yaml"
)
with open(nominal_param_path, "r") as yml:
    nominal_param = yaml.safe_load(yml)
optimization_param_path = (
    package_path["path"] + "/autoware_vehicle_adaptor/param/optimization_param.yaml"
)
with open(optimization_param_path, "r") as yml:
    optimization_param = yaml.safe_load(yml)
acc_queue_size = int(trained_model_param["trained_model_parameter"]["queue_size"]["acc_queue_size"])
steer_queue_size = int(trained_model_param["trained_model_parameter"]["queue_size"]["steer_queue_size"])
control_dt = 0.033
acc_delay_step = round(nominal_param["nominal_parameter"]["acceleration"]["acc_time_delay"] / control_dt)
steer_delay_step = round(nominal_param["nominal_parameter"]["steering"]["steer_time_delay"] / control_dt)
acc_time_constant = nominal_param["nominal_parameter"]["acceleration"]["acc_time_constant"]
steer_time_constant = nominal_param["nominal_parameter"]["steering"]["steer_time_constant"]
wheel_base = nominal_param["nominal_parameter"]["vehicle_info"]["wheel_base"]
vel_index = 0
acc_index = 1
steer_index = 2
prediction_step = 3
acc_input_indices_nom =  np.arange(3 + acc_queue_size -acc_delay_step, 3 + acc_queue_size -acc_delay_step + prediction_step)
steer_input_indices_nom = np.arange(3 + acc_queue_size + prediction_step + steer_queue_size -steer_delay_step, 3 + acc_queue_size + steer_queue_size -steer_delay_step + 2 * prediction_step)
prediction_step = int(optimization_param["optimization_parameter"]["setting"]["predict_step"])




def transform_to_sequence_data(X, seq_size, indices, prediction_step):
    X_seq = []
    start_num = 0
    for i in range(len(indices)):
        j = 0
        while start_num + j + prediction_step * seq_size <= indices[i]:
            X_seq.append(X[start_num + j + prediction_step * np.arange(seq_size)])
            j += 1
        start_num = indices[i]

    return np.array(X_seq)



def get_loss(criterion, model, X_batch, Y, Z, tanh_gain = 10, tanh_weight = 0.1, first_order_weight = 0.01, second_order_weight = 0.01):
    Y_pred, hc = model(X_batch,mode="get_lstm_states")
    # Calculate the loss
    loss = criterion(Y_pred, Y)
    # Calculate the integrated loss
    if integrate_states:
        predicted_states = X_batch[:, past_length, [vel_index, acc_index, steer_index]].unsqueeze(1)
        if integrate_yaw:
            predicted_yaw = Z[:,0,state_name_to_predicted_index["yaw"]]
        for i in range(integration_length):
            predicted_states_with_input_history = torch.cat((predicted_states, X_batch[:,[past_length + i], 3:]), dim=2)
            predicted_error, hc =  model(predicted_states_with_input_history, hc, mode="predict_with_hc")
            for j in range(prediction_step):
                predicted_states[:,0,0] = predicted_states[:,0,0] + predicted_states[:,0,1] * control_dt
                if integrate_yaw:
                    predicted_yaw = predicted_yaw + Z[:,i*prediction_step + j,state_name_to_predicted_index["vel"]]* torch.tan(predicted_states[:,0,2]) / wheel_base * control_dt
                    #predicted_yaw += predicted_states[:,:,0].clone()* torch.tan(predicted_states[:,:,2].clone()) / wheel_base * control_dt
                predicted_states[:,0,1] = predicted_states[:,0,1] + (X_batch[:,past_length + i, acc_input_indices_nom[j]] - predicted_states[:,0,1]) / acc_time_constant * control_dt
                predicted_states[:,0,2] = predicted_states[:,0,2] + (X_batch[:,past_length + i, steer_input_indices_nom[j]] - predicted_states[:,0,2]) / steer_time_constant * control_dt
            if integrate_vel:
                predicted_states[:,0,0] = predicted_states[:,0,0] + predicted_error[:,0,state_name_to_predicted_index["vel"]] * control_dt * prediction_step
            if integrate_yaw:
                predicted_yaw = predicted_yaw + predicted_error[:,0,state_name_to_predicted_index["yaw"]] * control_dt * prediction_step
            predicted_states[:,0,-2] = predicted_states[:,0,-2] + predicted_error[:,0,-2] * control_dt * prediction_step
            predicted_states[:,0,-1] = predicted_states[:,0,-1] + predicted_error[:,0,-1] * control_dt * prediction_step
            if integrate_vel:
                loss += integration_weight*criterion(predicted_states[:,0,0], Z[:,(i+1)*prediction_step,state_name_to_predicted_index["vel"]])
            if integrate_yaw:
                loss += integration_weight*criterion(predicted_yaw, Z[:,(i+1)*prediction_step,state_name_to_predicted_index["yaw"]])
            loss += integration_weight*criterion(predicted_states[:,0,-2], Z[:,(i+1)*prediction_step,state_name_to_predicted_index["acc"]])
            loss += integration_weight*criterion(predicted_states[:,0,-1], Z[:,(i+1)*prediction_step,state_name_to_predicted_index["steer"]])
        """
        if integrate_vel:
            loss += integration_weight*criterion(predicted_states[:,0,0], Z[:,-1,state_name_to_predicted_index["vel"]])
        if integrate_yaw:
            loss += integration_weight*criterion(predicted_yaw, Z[:,-1,state_name_to_predicted_index["yaw"]])
        loss += integration_weight*criterion(predicted_states[:,0,-2], Z[:,-1,state_name_to_predicted_index["acc"]])
        loss += integration_weight*criterion(predicted_states[:,0,-1], Z[:,-1,state_name_to_predicted_index["steer"]])
        """
    # Calculate the tanh loss
    tanh_loss = tanh_weight * torch.mean(torch.abs(torch.tanh(tanh_gain * (Y_pred[:,:,-1]-Y[:,:,-1]))))
    # Calculate the first order loss
    first_order_loss = first_order_weight * criterion(Y_pred[:, 1:] - Y_pred[:, :-1], Y[:, 1:] - Y[:, :-1])
    # Calculate the second order loss
    second_order_loss = second_order_weight * criterion(Y_pred[:, 2:] - 2 * Y_pred[:, 1:-1] + Y_pred[:, :-2], Y[:, 2:] - 2 * Y[:, 1:-1] + Y[:, :-2])
    # Calculate the total loss
    total_loss = loss + tanh_loss + first_order_loss + second_order_loss
    return total_loss

def validate_in_batches(model, criterion, X_val, Y_val, Z_val, batch_size=10000):
    model.eval()
    val_loss = 0.0
    num_batches = (X_val.size(0) + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, X_val.size(0))
            
            X_batch = X_val[start_idx:end_idx]
            Y_batch = Y_val[start_idx:end_idx]
            Z_batch = Z_val[start_idx:end_idx]
            
            loss = get_loss(criterion, model, X_batch, Y_batch, Z_batch)
            val_loss += loss.item() * (end_idx - start_idx)
    
    val_loss /= X_val.size(0)
    return val_loss
class EarlyStopping:
    """Class for early stopping in NN training."""

    def __init__(self, initial_loss, tol=0.01, patience=30):
        self.epoch = 0  # Initialise the counter for the number of epochs being monitored.
        self.best_loss = float("inf")  # Initialise loss of comparison with infinity 'inf'.
        self.patience = patience  # Initialise the number of epochs to be monitored with a parameter
        self.initial_loss = initial_loss
        self.tol = tol

    def __call__(self, current_loss):
        current_loss_num = current_loss
        if current_loss_num + self.tol * self.initial_loss > self.best_loss:
            self.epoch += 1
        else:
            self.epoch = 0
        if current_loss_num < self.best_loss:
            self.best_loss = current_loss_num
        if self.epoch >= self.patience:
            return True
        return False

    def reset(self):
        self.epoch = 0


class train_error_prediction_NN(add_data_from_csv.add_data_from_csv):
    """Class for training the error prediction NN."""

    def __init__(self, max_iter=10000, tol=1e-5, alpha_1=0.1**7, alpha_2=0.1**7):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_iter = max_iter
        self.tol = tol
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2

    def train_model(
        self,
        model: error_prediction_NN.ErrorPredictionNN,
        X_train: torch.Tensor,
        Y_train: torch.Tensor,
        Z_train: torch.Tensor,
        batch_size: int,
        learning_rates: list,
        patience: int,
        X_val: torch.Tensor,
        Y_val: torch.Tensor,
        Z_val: torch.Tensor,
        fix_lstm: bool = False,
    ):
        """Train the error prediction NN."""

        print("sample_size: ", X_train.shape[0] + X_val.shape[0])
        print("patience: ", patience)
        # Define the loss function.
        criterion = nn.L1Loss()
        # Fix the LSTM
        if fix_lstm:
            self.fix_lstm(model)
        # Define the optimizer.
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rates[0])
        # Define the initial loss.
        initial_loss = validate_in_batches(model,criterion,X_val, Y_val[:,past_length:], Z_val)
        print("initial_loss: ", initial_loss)
        # Define the early stopping object.
        early_stopping = EarlyStopping(initial_loss, tol=self.tol, patience=patience)
        # Data Loader
        train_dataset = DataLoader(
            TensorDataset(X_train, Y_train, Z_train), batch_size=batch_size, shuffle=True
        )
        # learning_rate index
        learning_rate_index = 0
        # Print learning rate
        print("learning rate: ", learning_rates[learning_rate_index])
        # Train the model.
        for i in range(self.max_iter):
            model.train()

            for X_batch, Y_batch, Z_batch in train_dataset:
                optimizer.zero_grad()
                # outputs = model(X_batch)
                loss = get_loss(criterion, model, X_batch, Y_batch[:,past_length:],Z_batch)
                for w in model.parameters():
                    loss += self.alpha_1 * torch.norm(w, 1) + self.alpha_2 * torch.norm(w, 2) ** 2
                loss.backward()
                optimizer.step()
            model.eval()
            val_loss = validate_in_batches(model,criterion,X_val, Y_val[:,past_length:], Z_val)
            if i % 10 == 1:
                print(val_loss, i)
            if early_stopping(val_loss):
                learning_rate_index += 1
                if learning_rate_index >= len(learning_rates):
                    break
                else:
                    print("update learning rate to ", learning_rates[learning_rate_index])
                    optimizer = torch.optim.Adam(
                        model.parameters(), lr=learning_rates[learning_rate_index]
                    )
                    early_stopping.reset()

    def get_trained_model(self, learning_rates=[1e-3, 1e-4, 1e-5, 1e-6], patience=10, batch_size=5):
        # Define Time Series Data
        X_train_np = transform_to_sequence_data(
            np.array(self.X_train_list)[:, 3:],
            past_length + prediction_length,
            self.division_indices_train,
            prediction_step,
        )
        Y_train_np = transform_to_sequence_data(
            np.array(self.Y_train_list)[:, state_component_predicted_index],
            past_length + prediction_length,
            self.division_indices_train,
            prediction_step,
        )
        Z_train_np = transform_to_sequence_data(
            np.array(self.Z_train_list)[:, state_component_predicted_index],
            (past_length + prediction_length) * prediction_step,
            self.division_indices_train,
            1,
        )[:, past_length * prediction_step: (past_length + integration_length) * prediction_step + 1]
        # rotation
        # Z_train_np[:,:, 0]
        X_val_np = transform_to_sequence_data(
            np.array(self.X_val_list)[:, 3:],
            past_length + prediction_length,
            self.division_indices_val,
            prediction_step,
        )
        Y_val_np = transform_to_sequence_data(
            np.array(self.Y_val_list)[:, state_component_predicted_index],
            past_length + prediction_length,
            self.division_indices_val,
            prediction_step,
        )
        Z_val_np = transform_to_sequence_data(
            np.array(self.Z_val_list)[:, state_component_predicted_index],
            (past_length + prediction_length) * prediction_step,
            self.division_indices_val,
            1,
        )[:, past_length * prediction_step: (past_length + integration_length) * prediction_step + 1]
        self.model = error_prediction_NN.ErrorPredictionNN(
            states_size=3, vel_index=0, output_size=len(state_component_predicted_index), prediction_length=prediction_length
        ).to(self.device)
        self.train_model(
            self.model,
            torch.tensor(X_train_np, dtype=torch.float32, device=self.device),
            torch.tensor(Y_train_np, dtype=torch.float32, device=self.device),
            torch.tensor(Z_train_np, dtype=torch.float32, device=self.device),
            batch_size,
            learning_rates,
            patience,
            torch.tensor(X_val_np, dtype=torch.float32, device=self.device),
            torch.tensor(Y_val_np, dtype=torch.float32, device=self.device),
            torch.tensor(Z_val_np, dtype=torch.float32, device=self.device),
        )
    def get_trained_ensemble_models(self, learning_rates=[1e-3, 1e-4, 1e-5, 1e-6], patience=10, batch_size=5, ensemble_size=5):
        # Define Time Series Data
        X_train_np = transform_to_sequence_data(
            np.array(self.X_train_list)[:, 3:],
            past_length + prediction_length,
            self.division_indices_train,
            prediction_step,
        )
        Y_train_np = transform_to_sequence_data(
            np.array(self.Y_train_list)[:, state_component_predicted_index],
            past_length + prediction_length,
            self.division_indices_train,
            prediction_step,
        )
        Z_train_np = transform_to_sequence_data(
            np.array(self.Z_train_list)[:, state_component_predicted_index],
            (past_length + prediction_length) * prediction_step,
            self.division_indices_train,
            1,
        )[:, past_length * prediction_step: (past_length + integration_length) * prediction_step + 1]
        # rotation
        # Z_train_np[:,:, 0]
        X_val_np = transform_to_sequence_data(
            np.array(self.X_val_list)[:, 3:],
            past_length + prediction_length,
            self.division_indices_val,
            prediction_step,
        )
        Y_val_np = transform_to_sequence_data(
            np.array(self.Y_val_list)[:, state_component_predicted_index],
            past_length + prediction_length,
            self.division_indices_val,
            prediction_step,
        )
        Z_val_np = transform_to_sequence_data(
            np.array(self.Z_val_list)[:, state_component_predicted_index],
            (past_length + prediction_length) * prediction_step,
            self.division_indices_val,
            1,
        )[:, past_length * prediction_step: (past_length + integration_length) * prediction_step + 1]
        self.model = error_prediction_NN.ErrorPredictionNN(
            states_size=3, vel_index=0, output_size=len(state_component_predicted_index), prediction_length=prediction_length
        ).to(self.device)
        self.train_model(
            self.model,
            torch.tensor(X_train_np, dtype=torch.float32, device=self.device),
            torch.tensor(Y_train_np, dtype=torch.float32, device=self.device),
            torch.tensor(Z_train_np, dtype=torch.float32, device=self.device),
            batch_size,
            learning_rates,
            patience,
            torch.tensor(X_val_np, dtype=torch.float32, device=self.device),
            torch.tensor(Y_val_np, dtype=torch.float32, device=self.device),
            torch.tensor(Z_val_np, dtype=torch.float32, device=self.device),
        )
        self.models = []
        for i in range(ensemble_size):
            temp_model = copy.deepcopy(self.model)
            self.train_model(temp_model,
                torch.tensor(X_train_np, dtype=torch.float32, device=self.device),
                torch.tensor(Y_train_np, dtype=torch.float32, device=self.device),
                torch.tensor(Z_train_np, dtype=torch.float32, device=self.device),
                batch_size,
                learning_rates,
                patience,
                torch.tensor(X_val_np, dtype=torch.float32, device=self.device),
                torch.tensor(Y_val_np, dtype=torch.float32, device=self.device),
                torch.tensor(Z_val_np, dtype=torch.float32, device=self.device),
                fix_lstm=True
            )
            self.models.append(temp_model)
    def update_saved_model(
        self, path, learning_rates=[1e-4, 1e-5, 1e-6], patience=10, batch_size=5
    ):
        X_train_np = transform_to_sequence_data(
            np.array(self.X_train_list)[:, 3:],
            past_length + prediction_length,
            self.division_indices_train,
            prediction_step,
        )
        Y_train_np = transform_to_sequence_data(
            np.array(self.Y_train_list)[:, -2:],
            past_length + prediction_length,
            self.division_indices_train,
            prediction_step,
        )
        Z_train_np = transform_to_sequence_data(
            np.array(self.Z_train_list)[:, -2:],
            past_length + prediction_length,
            self.division_indices_train,
            1,
        )[:, past_length * prediction_step: (past_length + integration_length) * prediction_step + 1]

        X_val_np = transform_to_sequence_data(
            np.array(self.X_val_list)[:, 3:],
            past_length + prediction_length,
            self.division_indices_val,
            prediction_step,
        )
        Y_val_np = transform_to_sequence_data(
            np.array(self.Y_val_list)[:, -2:],
            past_length + prediction_length,
            self.division_indices_val,
            prediction_step,
        )
        Z_val_np = transform_to_sequence_data(
            np.array(self.Z_val_list)[:, -2:],
            past_length + prediction_length,
            self.division_indices_val,
            1,
        )[:, past_length * prediction_step: (past_length + integration_length) * prediction_step + 1]
        self.model = torch.load(path)
        self.model.to(self.device)
        self.train_model(
            self.model,
            torch.tensor(X_train_np, dtype=torch.float32, device=self.device),
            torch.tensor(Y_train_np, dtype=torch.float32, device=self.device),
            torch.tensor(Z_train_np, dtype=torch.float32, device=self.device),
            batch_size,
            learning_rates,
            patience,
            torch.tensor(X_val_np, dtype=torch.float32, device=self.device),
            torch.tensor(Y_val_np, dtype=torch.float32, device=self.device),
            torch.tensor(Z_val_np, dtype=torch.float32, device=self.device),
        )

    def save_model(self, path="vehicle_model.pth"):
        self.model.to("cpu")
        torch.save(self.model, path)
    def save_ensemble_models(self, paths):
        for i in range(len(paths)):
            temp_model = self.models[i]
            temp_model.to("cpu")
            torch.save(temp_model, paths[i]) 
    def fix_lstm(self,model,randomize=0.01,):
        for param in model.acc_layer_1.parameters():
            param.requires_grad = False
        for param in model.steer_layer_1.parameters():
            param.requires_grad = False
        for param in model.acc_layer_2.parameters():
            param.requires_grad = False
        for param in model.steer_layer_2.parameters():
            param.requires_grad = False
        for param in model.lstm.parameters():
            param.requires_grad = False
        lb = -randomize
        ub = randomize
        nn.init.uniform_(model.complimentary_layer[0].weight, a=lb, b=ub)
        nn.init.uniform_(model.complimentary_layer[0].bias, a=lb, b=ub)
        nn.init.uniform_(model.linear_relu[0].weight, a=lb, b=ub)
        nn.init.uniform_(model.linear_relu[0].bias, a=lb, b=ub)
        nn.init.uniform_(model.final_layer.weight, a=lb, b=ub)
        nn.init.uniform_(model.final_layer.bias, a=lb, b=ub)  
