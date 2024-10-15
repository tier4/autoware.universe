from autoware_vehicle_adaptor.training import add_data_from_csv
from autoware_vehicle_adaptor.training import error_prediction_NN
from autoware_vehicle_adaptor.training import convert_model_to_csv
from autoware_vehicle_adaptor.training.early_stopping import EarlyStopping
from autoware_vehicle_adaptor.param import parameters
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.utils.data import WeightedRandomSampler
from pathlib import Path
import json
import copy
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import yaml
import random
import os
import types
from sklearn.linear_model import Lasso
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
prediction_step = int(optimization_param["optimization_parameter"]["setting"]["predict_step"])
acc_input_indices_nom =  np.arange(3 + acc_queue_size -acc_delay_step, 3 + acc_queue_size -acc_delay_step + prediction_step)
steer_input_indices_nom = np.arange(3 + acc_queue_size + prediction_step + steer_queue_size -steer_delay_step, 3 + acc_queue_size + steer_queue_size -steer_delay_step + 2 * prediction_step)




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



def get_loss(criterion, model, X_batch, Y, Z, adaptive_weight,tanh_gain = 10, tanh_weight = 0.1, first_order_weight = 0.01, second_order_weight = 0.01, randomize_previous_error=[0.5,0.1], integral_prob=0.0, alpha_jacobian=0.01, calc_jacobian_len=10, eps=1e-6):
    randomize_scale_tensor = torch.tensor(randomize_previous_error).to(X_batch.device)
    previous_error = Y[:, :past_length, -2:] + torch.mean(torch.abs(Y[:, :past_length, -2:]),dim=1).unsqueeze(1) * randomize_scale_tensor * torch.randn_like(Y[:, :past_length, -2:])
    Y_pred, hc = model(X_batch, previous_error=previous_error, mode="get_lstm_states")
    # Calculate the loss
    loss = criterion(Y_pred * adaptive_weight, Y[:,past_length:] * adaptive_weight)
    if alpha_jacobian is not None:
        hc_perturbed = model(X_batch[:, :past_length] + eps * torch.randn_like(X_batch[:, :past_length]),previous_error=previous_error + eps * torch.randn_like(previous_error), mode="only_encoder")
        Y_perturbed, _ = model(X_batch[:, past_length: past_length + calc_jacobian_len] + eps * torch.randn_like(X_batch[:, past_length: past_length + calc_jacobian_len]), hc=hc_perturbed, mode="predict_with_hc")
        loss += alpha_jacobian * criterion((Y_perturbed - Y_pred[:,:calc_jacobian_len]) / eps, torch.zeros_like(Y_perturbed))
    # Calculate the integrated loss
    #if integrate_states:
    if random.random() < integral_prob:
        predicted_states = X_batch[:, past_length, [vel_index, acc_index, steer_index]].unsqueeze(1)
        if integrate_yaw:
            predicted_yaw = Z[:,0,state_name_to_predicted_index["yaw"]]
        for i in range(integration_length):
            predicted_states_with_input_history = torch.cat((predicted_states, X_batch[:,[past_length + i], 3:]), dim=2)
            predicted_error, hc =  model(predicted_states_with_input_history, hc=hc, mode="predict_with_hc")
            for j in range(prediction_step):
                if integrate_vel:
                    predicted_states[:,0,0] = predicted_states[:,0,0] + predicted_states[:,0,1] * control_dt
                if integrate_yaw:
                    predicted_yaw = predicted_yaw + Z[:,i*prediction_step + j,state_name_to_predicted_index["vel"]]* torch.tan(predicted_states[:,0,2]) / wheel_base * control_dt
                predicted_states[:,0,1] = predicted_states[:,0,1] + (X_batch[:,past_length + i, acc_input_indices_nom[j]] - predicted_states[:,0,1]) * (1 - np.exp(- control_dt / acc_time_constant))
                predicted_states[:,0,2] = predicted_states[:,0,2] + (X_batch[:,past_length + i, steer_input_indices_nom[j]] - predicted_states[:,0,2])  * (1 - np.exp(- control_dt / steer_time_constant))
            if integrate_vel:
                predicted_states[:,0,0] = predicted_states[:,0,0] + predicted_error[:,0,state_name_to_predicted_index["vel"]] * control_dt * prediction_step
            else:
                predicted_states[:,0,0] = X_batch[:,past_length + i + 1, vel_index]
            if integrate_yaw:
                predicted_yaw = predicted_yaw + predicted_error[:,0,state_name_to_predicted_index["yaw"]] * control_dt * prediction_step
            predicted_states[:,0,-2] = predicted_states[:,0,-2] + predicted_error[:,0,-2] * control_dt * prediction_step
            predicted_states[:,0,-1] = predicted_states[:,0,-1] + predicted_error[:,0,-1] * control_dt * prediction_step
        if integrate_vel:
            loss += integration_weight*criterion(predicted_states[:,0,0], Z[:,-1,state_name_to_predicted_index["vel"]])
        if integrate_yaw:
            loss += integration_weight*criterion(predicted_yaw, Z[:,-1,state_name_to_predicted_index["yaw"]])
        loss += adaptive_weight[-2] * integration_weight*criterion(predicted_states[:,0,-2], Z[:,-1,state_name_to_predicted_index["acc"]])
        loss += adaptive_weight[-1] * integration_weight*criterion(predicted_states[:,0,-1], Z[:,-1,state_name_to_predicted_index["steer"]])
    # Calculate the tanh loss
    loss += tanh_weight * criterion(torch.tanh(tanh_gain * (Y_pred[:,:,-1]-Y[:,past_length:,-1])),torch.zeros_like(Y_pred[:,:,-1]))
    # Calculate the first order loss
    first_order_loss = first_order_weight * criterion(Y_pred[:, 1:] - Y_pred[:, :-1], torch.zeros_like(Y_pred[:, 1:]))
    # Calculate the second order loss
    second_order_loss = second_order_weight * criterion(Y_pred[:, 2:] - 2 * Y_pred[:, 1:-1] + Y_pred[:, :-2], torch.zeros_like(Y_pred[:, 2:]))
    # Calculate the total loss
    total_loss = loss + first_order_loss + second_order_loss
    return total_loss

def validate_in_batches(model, criterion, X_val, Y_val, Z_val, adaptive_weight, randomize_previous_error=[0.03,0.03],batch_size=10000,alpha_jacobian=None,calc_jacobian_len=10,eps=1e-5):
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
            
            loss = get_loss(criterion, model, X_batch, Y_batch, Z_batch,adaptive_weight=adaptive_weight,randomize_previous_error=randomize_previous_error,alpha_jacobian=alpha_jacobian,calc_jacobian_len=calc_jacobian_len,eps=eps)
            val_loss += loss.item() * (end_idx - start_idx)
    
    val_loss /= X_val.size(0)
    return val_loss
def get_each_component_loss(model, X_val, Y_val,tanh_gain = 10, tanh_weight = 0.1, first_order_weight = 0.01, second_order_weight = 0.01, batch_size=10000, window_size=10):
    model.eval()
    val_loss = np.zeros(Y_val.size(2))
    tanh_loss = 0.0
    first_order_loss = 0.0
    second_order_loss = 0.0
    num_batches = (X_val.size(0) + batch_size - 1) // batch_size
    Y_pred_list = []
    with torch.no_grad():
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, X_val.size(0))
            
            X_batch = X_val[start_idx:end_idx]
            Y_batch = Y_val[start_idx:end_idx]
            Y_pred, _ = model(X_batch, previous_error=Y_batch[:, :past_length, -2:], mode="get_lstm_states")
            Y_pred_list.append(Y_pred[:,window_size])
            # Calculate the loss
            loss = torch.mean(torch.abs(Y_pred - Y_batch[:,past_length:]),dim=(0,1))
            val_loss += loss.cpu().numpy() * (end_idx - start_idx)
            tanh_loss += tanh_weight * torch.mean(torch.abs(torch.tanh(tanh_gain * (Y_pred[:,:,-1]-Y_batch[:,past_length:,-1])))).item() * (end_idx - start_idx)
            first_order_loss += first_order_weight * torch.mean(torch.abs((Y_pred[:, 1:] - Y_pred[:, :-1]) - (Y_batch[:, past_length + 1:] - Y_batch[:, past_length:-1]))).item() * (end_idx - start_idx)
            second_order_loss += second_order_weight * torch.mean(torch.abs((Y_pred[:, 2:] - 2 * Y_pred[:, 1:-1] + Y_pred[:, :-2]) - (Y_batch[:, past_length + 2:] - 2 * Y_batch[:, past_length + 1:-1] + Y_batch[:, past_length:-2]))).item() * (end_idx - start_idx)
    val_loss /= (X_val.size(0) * Y_val.size(2))
    tanh_loss /= X_val.size(0)
    first_order_loss /= X_val.size(0)
    second_order_loss /= X_val.size(0)
    Y_pred_np = torch.cat(Y_pred_list,dim=0).cpu().detach().numpy()
    return np.concatenate((val_loss, [tanh_loss, first_order_loss, second_order_loss])), Y_pred_np


class train_error_prediction_NN(add_data_from_csv.add_data_from_csv):
    """Class for training the error prediction NN."""

    def __init__(self, max_iter=10000, tol=1e-5, alpha_1=0.1**7, alpha_2=0.1**7, alpha_jacobian=0.1**4):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_iter = max_iter
        self.tol = tol
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.alpha_jacobian = alpha_jacobian
        self.models = None
        self.weights_for_dataloader = None

        self.past_length = past_length
        self.prediction_length = prediction_length
        self.prediction_step = prediction_step
        self.acc_queue_size = acc_queue_size
        self.steer_queue_size = steer_queue_size
        self.adaptive_weight = torch.ones(len(state_component_predicted_index)).to(self.device)
    def train_model(
        self,
        model: error_prediction_NN.ErrorPredictionNN,
        X_train: torch.Tensor,
        Y_train: torch.Tensor,
        Z_train: torch.Tensor,
        batch_sizes: list,
        learning_rates: list,
        patience: int,
        X_val: torch.Tensor,
        Y_val: torch.Tensor,
        Z_val: torch.Tensor,
        fix_lstm: bool = False,
        integration_prob: float = 0.1
    ):
        """Train the error prediction NN."""

        print("sample_size: ", X_train.shape[0] + X_val.shape[0])
        print("patience: ", patience)
        # Define the loss function.
        criterion = nn.L1Loss()
        # Fix the LSTM
        if fix_lstm:
            self.fix_lstm(model)
        # save the original adaptive weight
        original_adaptive_weight = self.adaptive_weight.clone()
        print("original_adaptive_weight: ", original_adaptive_weight)
        # Define the optimizer.
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rates[0])
        # Define the initial loss.
        initial_loss = validate_in_batches(model,criterion,X_val, Y_val, Z_val, adaptive_weight=self.adaptive_weight)
        print("initial_loss: ", initial_loss)
        batch_size = batch_sizes[0]
        print("batch_size: ", batch_size)
        # Define the early stopping object.
        early_stopping = EarlyStopping(initial_loss, tol=self.tol, patience=patience)
        # Data Loader
        if self.weights_for_dataloader is None:
            weighted_sampler = None
            train_dataset = DataLoader(
                TensorDataset(X_train, Y_train, Z_train), batch_size=batch_size, shuffle=True
            )
        else:
            weighted_sampler = WeightedRandomSampler(weights=self.weights_for_dataloader, num_samples=len(self.weights_for_dataloader), replacement=True)
            train_dataset = DataLoader(
                TensorDataset(X_train, Y_train, Z_train), batch_size=batch_size, sampler=weighted_sampler
            )
        # learning_rate index
        learning_rate_index = 0
        # batch_size index
        # Print learning rate
        print("learning rate: ", learning_rates[learning_rate_index])
        # Train the model.
        for i in range(self.max_iter):
            model.train()

            for X_batch, Y_batch, Z_batch in train_dataset:
                optimizer.zero_grad()
                # outputs = model(X_batch)
                loss = get_loss(criterion, model, X_batch, Y_batch,Z_batch, adaptive_weight=self.adaptive_weight,integral_prob=integration_prob,alpha_jacobian=self.alpha_jacobian)
                for w in model.parameters():
                    loss += self.alpha_1 * torch.norm(w, 1) + self.alpha_2 * torch.norm(w, 2) ** 2
                loss.backward()
                optimizer.step()
            model.eval()
            val_loss = validate_in_batches(model,criterion,X_val, Y_val, Z_val,adaptive_weight=self.adaptive_weight)
            val_loss_with_original_weight = validate_in_batches(model,criterion,X_val, Y_val, Z_val,adaptive_weight=original_adaptive_weight)
            if i % 10 == 1:
                print("epoch: ", i)
                print("val_loss with original weight: ", val_loss_with_original_weight)
                print("val_loss: ", val_loss)
            if early_stopping(val_loss):
                learning_rate_index += 1
                batch_size = batch_sizes[min(learning_rate_index, len(batch_sizes) - 1)]
                if learning_rate_index >= len(learning_rates):
                    break
                batch_size = batch_sizes[min(learning_rate_index, len(batch_sizes) - 1)]
                optimizer = torch.optim.Adam(
                    model.parameters(), lr=learning_rates[learning_rate_index]
                )
                print("update learning rate to ", learning_rates[learning_rate_index])
                print("batch size:", batch_size)
                if self.weights_for_dataloader is None:
                    train_dataset = DataLoader(
                        TensorDataset(X_train, Y_train, Z_train), batch_size=batch_size, shuffle=True
                    )
                else:
                    train_dataset = DataLoader(
                        TensorDataset(X_train, Y_train, Z_train), batch_size=batch_size, sampler=weighted_sampler
                    )
                early_stopping.reset()
                if learning_rates[learning_rate_index - 1] < 3e-4:
                    self.update_adaptive_weight(model,X_train,Y_train)
                print("adaptive_weight: ", self.adaptive_weight)

    def relearn_model(
        self,
        model: error_prediction_NN.ErrorPredictionNN,
        X_train: torch.Tensor,
        Y_train: torch.Tensor,
        Z_train: torch.Tensor,
        batch_sizes: list,
        learning_rates: list,
        patience: int,
        X_val: torch.Tensor,
        Y_val: torch.Tensor,
        Z_val: torch.Tensor,
        fix_lstm: bool = False,
        integration_prob: float = 0.1,
        randomize: float = 0.001,
        X_test=None,
        Y_test=None,
        Z_test=None,
        plt_save_dir=None,
        window_size=10,
        save_path=None,
    ):
        print("randomize: ", randomize)
        self.update_adaptive_weight(model,X_train,Y_train)
        original_adaptive_weight = self.adaptive_weight.clone()
        criterion = nn.L1Loss()
        original_train_loss = validate_in_batches(model,criterion,X_train, Y_train, Z_train, adaptive_weight=original_adaptive_weight)
        original_each_component_train_loss, Y_train_pred_origin = get_each_component_loss(model, X_train, Y_train)
        original_val_loss = validate_in_batches(model,criterion,X_val, Y_val, Z_val, adaptive_weight=original_adaptive_weight)
        original_each_component_val_loss, Y_val_pred_origin = get_each_component_loss(model, X_val, Y_val)
        if X_test is not None:
            original_test_loss = validate_in_batches(model,criterion,X_test, Y_test, Z_test, adaptive_weight=original_adaptive_weight)
            original_each_component_test_loss, Y_test_pred_origin = get_each_component_loss(model, X_test, Y_test)
        relearned_model = copy.deepcopy(model)
        relearned_model.lstm_encoder.flatten_parameters()
        relearned_model.lstm.flatten_parameters()
        with torch.no_grad():
            if fix_lstm:
                relearned_model.complimentary_layer[0].weight += randomize * torch.randn_like(model.complimentary_layer[0].weight)
                relearned_model.complimentary_layer[0].bias += randomize * torch.randn_like(model.complimentary_layer[0].bias)
                relearned_model.linear_relu[0].weight += randomize * torch.randn_like(model.linear_relu[0].weight)
                relearned_model.linear_relu[0].bias += randomize * torch.randn_like(model.linear_relu[0].bias)
                relearned_model.final_layer.weight += randomize * torch.randn_like(model.final_layer.weight)
                relearned_model.final_layer.bias += randomize * torch.randn_like(model.final_layer.bias)
            else:
                for w in relearned_model.parameters():
                    w += randomize * torch.randn_like(w)
        self.train_model(
            relearned_model,
            X_train,
            Y_train,
            Z_train,
            batch_sizes,
            learning_rates,
            patience,
            X_val,
            Y_val,
            Z_val,
            fix_lstm=fix_lstm,
            integration_prob=integration_prob,
        )
        relearned_train_loss = validate_in_batches(relearned_model,criterion,X_train, Y_train, Z_train, adaptive_weight=original_adaptive_weight)
        relearned_each_component_train_loss, Y_train_pred_relearned = get_each_component_loss(relearned_model, X_train, Y_train,window_size=window_size)
        relearned_val_loss = validate_in_batches(relearned_model,criterion,X_val, Y_val, Z_val, adaptive_weight=original_adaptive_weight)
        relearned_each_component_val_loss, Y_val_pred_relearned = get_each_component_loss(relearned_model, X_val, Y_val,window_size=window_size)
        if X_test is not None:
            relearned_test_loss = validate_in_batches(relearned_model,criterion,X_test, Y_test, Z_test, adaptive_weight=original_adaptive_weight)
            relearned_each_component_test_loss, Y_test_pred_relearned = get_each_component_loss(relearned_model, X_test, Y_test,window_size=window_size)
        print("original_train_loss: ", original_train_loss)
        print("relearned_train_loss: ", relearned_train_loss)
        print("original_val_loss: ", original_val_loss)
        print("relearned_val_loss: ", relearned_val_loss)
        if X_test is not None:
            print("original_test_loss: ", original_test_loss)
            print("relearned_test_loss: ", relearned_test_loss)

        print("original_each_component_train_loss: ", original_each_component_train_loss)
        print("relearned_each_component_train_loss: ", relearned_each_component_train_loss)
        print("original_each_component_val_loss: ", original_each_component_val_loss)
        print("relearned_each_component_val_loss: ", relearned_each_component_val_loss)
        if X_test is not None:
            print("original_each_component_test_loss: ", original_each_component_test_loss)
            print("relearned_each_component_test_loss: ", relearned_each_component_test_loss)

        original_train_prediction, nominal_train_prediction = self.get_acc_steer_prediction(model,X_train.to("cpu"),Y_train.to("cpu"),window_size)
        relearned_train_prediction, _ = self.get_acc_steer_prediction(relearned_model,X_train.to("cpu"),Y_train.to("cpu"),window_size)

        original_val_prediction, nominal_val_prediction = self.get_acc_steer_prediction(model,X_val.to("cpu"),Y_val.to("cpu"),window_size)
        relearned_val_prediction, _ = self.get_acc_steer_prediction(relearned_model,X_val.to("cpu"),Y_val.to("cpu"),window_size)

        nominal_singed_train_prediction_error = X_train[:,past_length+window_size,[1,2]].to("cpu").detach().numpy() - nominal_train_prediction
        original_singed_train_prediction_error = X_train[:,past_length+window_size,[1,2]].to("cpu").detach().numpy() - original_train_prediction
        relearned_singed_train_prediction_error = X_train[:,past_length+window_size,[1,2]].to("cpu").detach().numpy() - relearned_train_prediction

        nominal_signed_val_prediction_error = X_val[:,past_length+window_size,[1,2]].to("cpu").detach().numpy() - nominal_val_prediction
        original_singed_val_prediction_error = X_val[:,past_length+window_size,[1,2]].to("cpu").detach().numpy() - original_val_prediction
        relearned_singed_val_prediction_error = X_val[:,past_length+window_size,[1,2]].to("cpu").detach().numpy() - relearned_val_prediction

        if X_test is not None:
            original_test_prediction, nominal_test_prediction = self.get_acc_steer_prediction(model,X_test.to("cpu"),Y_test.to("cpu"),window_size)
            relearned_test_prediction, _ = self.get_acc_steer_prediction(relearned_model,X_test.to("cpu"),Y_test.to("cpu"),window_size)

            nominal_signed_test_prediction_error = X_test[:,past_length+window_size,[1,2]].to("cpu").detach().numpy() - nominal_test_prediction
            original_singed_test_prediction_error = X_test[:,past_length+window_size,[1,2]].to("cpu").detach().numpy() - original_test_prediction
            relearned_singed_test_prediction_error = X_test[:,past_length+window_size,[1,2]].to("cpu").detach().numpy() - relearned_test_prediction

        print("nominal acc train prediction loss:", np.mean(np.abs(nominal_singed_train_prediction_error[:,0])))
        print("original acc train prediction loss:", np.mean(np.abs(original_singed_train_prediction_error[:,0])))
        print("relearned acc train prediction loss:", np.mean(np.abs(relearned_singed_train_prediction_error[:,0])))
        print("nominal steer train prediction loss:", np.mean(np.abs(nominal_singed_train_prediction_error[:,1])))
        print("original steer train prediction loss:", np.mean(np.abs(original_singed_train_prediction_error[:,1])))
        print("relearned steer train prediction loss:", np.mean(np.abs(relearned_singed_train_prediction_error[:,1])))
        print("nominal acc val prediction loss:", np.mean(np.abs(nominal_signed_val_prediction_error[:,0])))
        print("original acc val prediction loss:", np.mean(np.abs(original_singed_val_prediction_error[:,0])))
        print("relearned acc val prediction loss:", np.mean(np.abs(relearned_singed_val_prediction_error[:,0])))
        print("nominal steer val prediction loss:", np.mean(np.abs(nominal_signed_val_prediction_error[:,1])))
        print("original steer val prediction loss:", np.mean(np.abs(original_singed_val_prediction_error[:,1])))
        print("relearned steer val prediction loss:", np.mean(np.abs(relearned_singed_val_prediction_error[:,1])))
        if X_test is not None:
            print("nominal acc test prediction loss:", np.mean(np.abs(nominal_signed_test_prediction_error[:,0])))
            print("original acc test prediction loss:", np.mean(np.abs(original_singed_test_prediction_error[:,0])))
            print("relearned acc test prediction loss:", np.mean(np.abs(relearned_singed_test_prediction_error[:,0])))
            print("nominal steer test prediction loss:", np.mean(np.abs(nominal_signed_test_prediction_error[:,1])))
            print("original steer test prediction loss:", np.mean(np.abs(original_singed_test_prediction_error[:,1])))
            print("relearned steer test prediction loss:", np.mean(np.abs(relearned_singed_test_prediction_error[:,1])))
        if plt_save_dir is not None:
            if X_test is None:
                fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(24,15), tight_layout=True)
            else:
                fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(24,15), tight_layout=True)
            fig.suptitle("acc steer prediction error")
            axes[0,0].plot(nominal_singed_train_prediction_error[:,0],label="nominal")
            axes[0,0].plot(original_singed_train_prediction_error[:,0],label="original")
            axes[0,0].plot(relearned_singed_train_prediction_error[:,0],label="relearned")
            axes[0,0].scatter(np.arange(len(nominal_singed_train_prediction_error[:,0])),np.zeros(len(nominal_singed_train_prediction_error[:,0])), s=1)
            axes[0,0].set_title("acc prediction error for training data")
            axes[0,0].legend()
            axes[0,1].plot(nominal_signed_val_prediction_error[:,0],label="nominal")
            axes[0,1].plot(original_singed_val_prediction_error[:,0],label="original")
            axes[0,1].plot(relearned_singed_val_prediction_error[:,0],label="relearned")
            axes[0,1].scatter(np.arange(len(nominal_signed_val_prediction_error[:,0])),np.zeros(len(nominal_signed_val_prediction_error[:,0])), s=1)
            axes[0,1].set_title("acc prediction error for validation data")
            axes[0,1].legend()
            axes[1,0].plot(nominal_singed_train_prediction_error[:,1],label="nominal")
            axes[1,0].plot(original_singed_train_prediction_error[:,1],label="original")
            axes[1,0].plot(relearned_singed_train_prediction_error[:,1],label="relearned")
            axes[1,0].scatter(np.arange(len(nominal_singed_train_prediction_error[:,1])),np.zeros(len(nominal_singed_train_prediction_error[:,1])), s=1)
            axes[1,0].set_title("steer prediction error for training data")
            axes[1,0].legend()
            axes[1,1].plot(nominal_signed_val_prediction_error[:,1],label="nominal")
            axes[1,1].plot(original_singed_val_prediction_error[:,1],label="original")
            axes[1,1].plot(relearned_singed_val_prediction_error[:,1],label="relearned")
            axes[1,1].scatter(np.arange(len(nominal_signed_val_prediction_error[:,1])),np.zeros(len(nominal_signed_val_prediction_error[:,1])), s=1)
            axes[1,1].set_title("steer prediction error for validation data")
            axes[1,1].legend()
            if X_test is not None:
                axes[0,2].plot(nominal_signed_test_prediction_error[:,0],label="nominal")
                axes[0,2].plot(original_singed_test_prediction_error[:,0],label="original")
                axes[0,2].plot(relearned_singed_test_prediction_error[:,0],label="relearned")
                axes[0,2].scatter(np.arange(len(nominal_signed_test_prediction_error[:,0])),np.zeros(len(nominal_signed_test_prediction_error[:,0])), s=1)
                axes[0,2].set_title("acc prediction error for test data")
                axes[0,2].legend()
                axes[1,2].plot(nominal_signed_test_prediction_error[:,1],label="nominal")
                axes[1,2].plot(original_singed_test_prediction_error[:,1],label="original")
                axes[1,2].plot(relearned_singed_test_prediction_error[:,1],label="relearned")
                axes[1,2].scatter(np.arange(len(nominal_signed_test_prediction_error[:,1])),np.zeros(len(nominal_signed_test_prediction_error[:,1])), s=1)
                axes[1,2].set_title("steer prediction error for test data")
                axes[1,2].legend()
            if not os.path.isdir(plt_save_dir):
                os.mkdir(plt_save_dir)
            plt.savefig(plt_save_dir + "/acc_steer_prediction_error.png")
            plt.close()
            Y_nominal_train_pred = Y_train[:,past_length+window_size,:].to("cpu").detach().numpy()
            Y_nominal_val_pred = Y_val[:,past_length+window_size,:].to("cpu").detach().numpy()
            if X_test is None:
                fig, axes = plt.subplots(nrows=len(state_component_predicted), ncols=2, figsize=(24,15), tight_layout=True)
            else:
                Y_nominal_test_pred = Y_test[:,past_length+window_size,:].to("cpu").detach().numpy()
                fig, axes = plt.subplots(nrows=len(state_component_predicted), ncols=3, figsize=(24,15), tight_layout=True)
            fig.suptitle("each component error")
            for i in range(len(state_component_predicted)):
                axes[i,0].plot(Y_nominal_train_pred[:,i],label="nominal")
                axes[i,0].plot(Y_nominal_train_pred[:,i]-Y_train_pred_origin[:,i],label="original")
                axes[i,0].plot(Y_nominal_train_pred[:,i]-Y_train_pred_relearned[:,i],label="relearned")
                axes[i,0].scatter(np.arange(len(Y_nominal_train_pred[:,i])),np.zeros(len(Y_nominal_train_pred[:,i])), s=1)
                axes[i,0].set_title(state_component_predicted[i] + " error for training data")
                axes[i,0].legend()
                axes[i,1].plot(Y_nominal_val_pred[:,i],label="nominal")
                axes[i,1].plot(Y_nominal_val_pred[:,i]-Y_val_pred_origin[:,i],label="original")
                axes[i,1].plot(Y_nominal_val_pred[:,i]-Y_val_pred_relearned[:,i],label="relearned")
                axes[i,1].scatter(np.arange(len(Y_nominal_val_pred[:,i])),np.zeros(len(Y_nominal_val_pred[:,i])), s=1)
                axes[i,1].set_title(state_component_predicted[i] + " error for validation data")
                axes[i,1].legend()
                if X_test is not None:
                    axes[i,2].plot(Y_nominal_test_pred[:,i],label="nominal")
                    axes[i,2].plot(Y_nominal_test_pred[:,i]-Y_test_pred_origin[:,i],label="original")
                    axes[i,2].plot(Y_nominal_test_pred[:,i]-Y_test_pred_relearned[:,i],label="relearned")
                    axes[i,2].scatter(np.arange(len(Y_nominal_test_pred[:,i])),np.zeros(len(Y_nominal_test_pred[:,i])), s=4)
                    axes[i,2].set_title(state_component_predicted[i] + " error for test data")
                    axes[i,2].legend()
            plt.savefig(plt_save_dir + "/each_component_error.png")
            plt.close()
        if save_path is not None:
            self.save_given_model(relearned_model, save_path)
        if relearned_val_loss < original_val_loss:
            return relearned_model, True
        else:
            return model, False

    def get_trained_model(self, learning_rates=[1e-3, 1e-4, 1e-5, 1e-6], patience=10, batch_sizes=[100,10,100]):
        print("state_component_predicted: ", state_component_predicted)
        # Define Time Series Data
        X_train_np, Y_train_np, Z_train_np = self.get_sequence_data(self.X_train_list, self.Y_train_list, self.Z_train_list,self.division_indices_train)
        X_val_np, Y_val_np, Z_val_np = self.get_sequence_data(self.X_val_list, self.Y_val_list, self.Z_val_list,self.division_indices_val)
        self.model = error_prediction_NN.ErrorPredictionNN(
            states_size=3, vel_index=0, output_size=len(state_component_predicted_index), prediction_length=prediction_length
        ).to(self.device)
        self.update_adaptive_weight(None,torch.tensor(X_train_np, dtype=torch.float32, device=self.device),torch.tensor(Y_train_np, dtype=torch.float32, device=self.device))
        self.train_model(
            self.model,
            torch.tensor(X_train_np, dtype=torch.float32, device=self.device),
            torch.tensor(Y_train_np, dtype=torch.float32, device=self.device),
            torch.tensor(Z_train_np, dtype=torch.float32, device=self.device),
            batch_sizes,
            learning_rates,
            patience,
            torch.tensor(X_val_np, dtype=torch.float32, device=self.device),
            torch.tensor(Y_val_np, dtype=torch.float32, device=self.device),
            torch.tensor(Z_val_np, dtype=torch.float32, device=self.device),
        )
    def get_relearned_model(self, learning_rates=[1e-3, 1e-4, 1e-5, 1e-6], patience=10, batch_sizes=[100], randomize=0.001,plt_save_dir=None,save_path=None):
        self.model.to(self.device)
        # Define Time Series Data
        X_train_np, Y_train_np, Z_train_np = self.get_sequence_data(self.X_train_list, self.Y_train_list, self.Z_train_list,self.division_indices_train)
        X_val_np, Y_val_np, Z_val_np = self.get_sequence_data(self.X_val_list, self.Y_val_list, self.Z_val_list,self.division_indices_val)
        if len(self.X_test_list) > 0:
            X_test_np, Y_test_np, Z_test_np = self.get_sequence_data(self.X_test_list, self.Y_test_list, self.Z_test_list,self.division_indices_test)
            X_test = torch.tensor(X_test_np, dtype=torch.float32, device=self.device)
            Y_test = torch.tensor(Y_test_np, dtype=torch.float32, device=self.device)
            Z_test = torch.tensor(Z_test_np, dtype=torch.float32, device=self.device)
        else:
            X_test = None
            Y_test = None
            Z_test = None
            
        self.model, updated = self.relearn_model(
            self.model,
            torch.tensor(X_train_np, dtype=torch.float32, device=self.device),
            torch.tensor(Y_train_np, dtype=torch.float32, device=self.device),
            torch.tensor(Z_train_np, dtype=torch.float32, device=self.device),
            batch_sizes,
            learning_rates,
            patience,
            torch.tensor(X_val_np, dtype=torch.float32, device=self.device),
            torch.tensor(Y_val_np, dtype=torch.float32, device=self.device),
            torch.tensor(Z_val_np, dtype=torch.float32, device=self.device),
            randomize=randomize,
            X_test=X_test,
            Y_test=Y_test,
            Z_test=Z_test,
            plt_save_dir=plt_save_dir,
            save_path=save_path
        )
        return updated

    def get_trained_ensemble_models(self, learning_rates=[1e-3, 1e-4, 1e-5, 1e-6], patience=10, batch_sizes=[100,10,100], ensemble_size=5):
        print("state_component_predicted: ", state_component_predicted)
        # Define Time Series Data
        X_train_np, Y_train_np, Z_train_np = self.get_sequence_data(self.X_train_list, self.Y_train_list, self.Z_train_list,self.division_indices_train)    
        X_val_np, Y_val_np, Z_val_np = self.get_sequence_data(self.X_val_list, self.Y_val_list, self.Z_val_list,self.division_indices_val)
        self.model = error_prediction_NN.ErrorPredictionNN(
            states_size=3, vel_index=0, output_size=len(state_component_predicted_index), prediction_length=prediction_length
        ).to(self.device)
        print("______________________________")
        print("ensemble number: ", 0)
        print("______________________________")
        self.train_model(
            self.model,
            torch.tensor(X_train_np, dtype=torch.float32, device=self.device),
            torch.tensor(Y_train_np, dtype=torch.float32, device=self.device),
            torch.tensor(Z_train_np, dtype=torch.float32, device=self.device),
            batch_sizes,
            learning_rates,
            patience,
            torch.tensor(X_val_np, dtype=torch.float32, device=self.device),
            torch.tensor(Y_val_np, dtype=torch.float32, device=self.device),
            torch.tensor(Z_val_np, dtype=torch.float32, device=self.device),
        )
        self.models = [self.model]
        for i in range(ensemble_size - 1):
            print("______________________________")
            print("ensemble number: ", i + 1)
            print("______________________________")
            temp_model = copy.deepcopy(self.model)
            self.train_model(temp_model,
                torch.tensor(X_train_np, dtype=torch.float32, device=self.device),
                torch.tensor(Y_train_np, dtype=torch.float32, device=self.device),
                torch.tensor(Z_train_np, dtype=torch.float32, device=self.device),
                batch_sizes,
                learning_rates,
                patience,
                torch.tensor(X_val_np, dtype=torch.float32, device=self.device),
                torch.tensor(Y_val_np, dtype=torch.float32, device=self.device),
                torch.tensor(Z_val_np, dtype=torch.float32, device=self.device),
                fix_lstm=True
            )
            self.models.append(temp_model)
    def update_saved_model(
        self, path, learning_rates=[1e-4, 1e-5, 1e-6], patience=10, batch_sizes=[100,10,100]
    ):
        X_train_np, Y_train_np, Z_train_np = self.get_sequence_data(self.X_train_list, self.Y_train_list, self.Z_train_list,self.division_indices_train)
        X_val_np, Y_val_np, Z_val_np = self.get_sequence_data(self.X_val_list, self.Y_val_list, self.Z_val_list,self.division_indices_val)
        self.model = torch.load(path)
        self.model.to(self.device)
        self.train_model(
            self.model,
            torch.tensor(X_train_np, dtype=torch.float32, device=self.device),
            torch.tensor(Y_train_np, dtype=torch.float32, device=self.device),
            torch.tensor(Z_train_np, dtype=torch.float32, device=self.device),
            batch_sizes,
            learning_rates,
            patience,
            torch.tensor(X_val_np, dtype=torch.float32, device=self.device),
            torch.tensor(Y_val_np, dtype=torch.float32, device=self.device),
            torch.tensor(Z_val_np, dtype=torch.float32, device=self.device),
        )
    def get_sequence_data(self, X, Y, Z, division_indices, acc_threshold=2.0, steer_threshold=0.7, acc_change_threshold=1.5, steer_change_threshold=0.8, acc_change_window=10, steer_change_window=10):
        X_seq = transform_to_sequence_data(
            np.array(X)[:, 3:],
            past_length + prediction_length,
            division_indices,
            prediction_step,
        )
        Y_seq = transform_to_sequence_data(
            np.array(Y)[:, state_component_predicted_index],
            past_length + prediction_length,
            division_indices,
            prediction_step,
        )
        Z_seq = transform_to_sequence_data(
            np.array(Z)[:, state_component_predicted_index],
            (past_length + prediction_length) * prediction_step,
            division_indices,
            1,
        )[:, past_length * prediction_step: (past_length + integration_length) * prediction_step + 1]
        X_seq_filtered = []
        Y_seq_filtered = []
        Z_seq_filtered = []
        for i in range(X_seq.shape[0]):
            acc = X_seq[i, :, 1]
            steer = X_seq[i, :, 2]
            acc_input = X_seq[i, :, acc_input_indices_nom]
            steer_input = X_seq[i, :, steer_input_indices_nom]
            acc_change = (acc[acc_change_window:] - acc[:-acc_change_window]) / (acc_change_window * control_dt)
            steer_change = (steer[steer_change_window:] - steer[:-steer_change_window]) / (steer_change_window * control_dt)
            if (
                (np.abs(acc).max() < acc_threshold) and
                (np.abs(steer).max() < steer_threshold) and
                (np.abs(acc_input).max() < acc_threshold) and
                (np.abs(steer_input).max() < steer_threshold) and
                (np.abs(acc_change).max() < acc_change_threshold) and
                (np.abs(steer_change).max() < steer_change_threshold)
            ):
                X_seq_filtered.append(X_seq[i])
                Y_seq_filtered.append(Y_seq[i])
                Z_seq_filtered.append(Z_seq[i])
        return np.array(X_seq_filtered), np.array(Y_seq_filtered), np.array(Z_seq_filtered)

    def save_model(self, path="vehicle_model.pth"):
        self.model.to("cpu")
        torch.save(self.model, path)
        save_dir = path.replace(".pth", "")
        convert_model_to_csv.convert_model_to_csv(self.model, save_dir, state_component_predicted)
    def save_given_model(self, model, path="vehicle_model.pth"):
        model.to("cpu")
        torch.save(model, path)
        save_dir = path.replace(".pth", "")
        convert_model_to_csv.convert_model_to_csv(model, save_dir, state_component_predicted)
    def save_ensemble_models(self, paths):
        for i in range(len(paths)):
            temp_model = self.models[i]
            temp_model.to("cpu")
            torch.save(temp_model, paths[i])
            save_dir = paths[i].replace(".pth", "")
            convert_model_to_csv.convert_model_to_csv(temp_model, save_dir, state_component_predicted)
    def fix_lstm(self,model,randomize=0.001,):
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
        
        #lb = -randomize
        #ub = randomize
        #nn.init.uniform_(model.complimentary_layer[0].weight, a=lb, b=ub)
        #nn.init.uniform_(model.complimentary_layer[0].bias, a=lb, b=ub)
        #nn.init.uniform_(model.linear_relu[0].weight, a=lb, b=ub)
        #nn.init.uniform_(model.linear_relu[0].bias, a=lb, b=ub)
        #nn.init.uniform_(model.final_layer.weight, a=lb, b=ub)
        #nn.init.uniform_(model.final_layer.bias, a=lb, b=ub)
        with torch.no_grad():
            model.complimentary_layer[0].weight += randomize * torch.randn_like(model.complimentary_layer[0].weight)
            model.complimentary_layer[0].bias += randomize * torch.randn_like(model.complimentary_layer[0].bias)
            model.linear_relu[0].weight += randomize * torch.randn_like(model.linear_relu[0].weight)
            model.linear_relu[0].bias += randomize * torch.randn_like(model.linear_relu[0].bias)
            model.final_layer.weight += randomize * torch.randn_like(model.final_layer.weight)
            model.final_layer.bias += randomize * torch.randn_like(model.final_layer.bias)
        

    def extract_features_from_data(self,X,division_indices,window_size=10):
        X_train_np, _, _ = self.get_sequence_data(X, self.Y_train_list, self.Z_train_list,division_indices)
        X_extracted = []
        for i in range(X_train_np.shape[0]):
            X_tmp = X_train_np[i][past_length:past_length + window_size]
            vel = X_tmp[:, 0].mean()
            acc = X_tmp[:, 1].mean()
            steer = X_tmp[:, 2].mean()
            acc_change = X_tmp[:, 1].max() - X_tmp[:, 1].min()
            steer_change = X_tmp[:, 2].max() - X_tmp[:, 2].min()
            X_extracted.append([vel, acc, steer, acc_change, steer_change])
        return X_extracted
    def extract_features_for_trining_data(self,window_size=10, vel_scale=0.1,acc_scale=1.0,steer_scale=1.5,acc_change_scale=5.0,steer_change_scale=5.0,bandwidth=0.3):
        self.X_extracted = np.array(self.extract_features_from_data(self.X_train_list,self.division_indices_train,window_size))
        self.scaling = [vel_scale, acc_scale, steer_scale, acc_change_scale, steer_change_scale]
        self.X_extracted_scaled = self.scaling * self.X_extracted
        self.kde_acc = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(self.X_extracted_scaled[:,[0,1,3]])
        self.kde_steer = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(self.X_extracted_scaled[:,[0,2,4]])
        self.kde_acc_steer = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(self.X_extracted_scaled)

    def plot_extracted_features(self,show_flag=True,save_dir=None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.X_extracted[:,0],self.X_extracted[:,1],self.X_extracted[:,3])
        ax.set_xlabel('vel')
        ax.set_ylabel('acc')
        ax.set_zlabel('acc_change')
        ax.set_title('vel acc acc_change')
        if show_flag:
            plt.show()
        if save_dir:
            plt.savefig(save_dir + "/vel_acc_acc_change.png")
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.X_extracted[:,0],self.X_extracted[:,2],self.X_extracted[:,4])
        ax.set_xlabel('vel')
        ax.set_ylabel('steer')
        ax.set_zlabel('steer_change')
        ax.set_title('vel steer steer_change')
        if show_flag:
            plt.show()
        if save_dir:
            plt.savefig(save_dir + "/vel_steer_steer_change.png")
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(24,15), tight_layout=True)
        fig.suptitle("sequential plot")
        axes[0,0].plot(self.X_extracted[:,1],label="acc")
        axes[0,0].set_title("acc")
        axes[0,0].legend()
        axes[0,1].plot(self.X_extracted[:,3],label="acc_change")
        axes[0,1].set_title("acc_change")
        axes[0,1].legend()
        axes[1,0].plot(self.X_extracted[:,2],label="steer")
        axes[1,0].set_title("steer")
        axes[1,0].legend()
        axes[1,1].plot(self.X_extracted[:,4],label="steer_change")
        axes[1,1].set_title("steer_change")
        axes[1,1].legend()
        if show_flag:
            plt.show()
        if save_dir:
            plt.savefig(save_dir + "/extracted_data_sequential_plot.png")
    def calc_prediction_error_and_std(self, window_size=10, error_smoothing_window=100, show_density=False):
        if self.models is None:
            print("models are not trained")
            return
        X_test_np, Y_test_np, _ = self.get_sequence_data(self.X_test_list, self.Y_test_list, self.Z_test_list,self.division_indices_test)
        result_dict = {}
        nominal_prediction = []
        prediction = []
        for model in self.models:
            prediction_tmp, nominal_prediction = self.get_acc_steer_prediction(model,torch.tensor(X_test_np, dtype=torch.float32),torch.tensor(Y_test_np, dtype=torch.float32),window_size)
            
            prediction.append(prediction_tmp)
        result_dict["prediction_by_models"] = prediction
        mean_prediction = np.mean(prediction,axis=0)
        std_prediction = np.std(prediction,axis=0)
        result_dict["mean_prediction"] = mean_prediction
        result_dict["std_prediction"] = std_prediction
        result_dict["nominal_prediction"] = nominal_prediction
        result_dict["true_value"] = X_test_np[:,past_length+window_size,[1,2]]
        signed_prediction_error = X_test_np[:,past_length+window_size,[1,2]] - mean_prediction
        signed_nominal_prediction_error = X_test_np[:,past_length+window_size,[1,2]] - nominal_prediction
        if show_density:
            X_test_extracted = np.array(self.extract_features_from_data(self.X_test_list,self.division_indices_test,window_size))
            X_test_extracted_scaled = self.scaling * X_test_extracted
            prob_acc = np.exp(self.kde_acc.score_samples(X_test_extracted_scaled[:,[0,1,3]]))
            prob_steer = np.exp(self.kde_steer.score_samples(X_test_extracted_scaled[:,[0,2,4]]))
            prob_acc_steer = np.exp(self.kde_acc_steer.score_samples(X_test_extracted_scaled))
            w = np.ones(error_smoothing_window) / error_smoothing_window
            residual_acc_error_ratio = np.convolve(np.abs(signed_prediction_error[:,0]),w,mode="same") / (np.convolve(np.abs(signed_nominal_prediction_error[:,0]),w,mode="same") + 1e-3)
            residual_steer_error_ratio = np.convolve(np.abs(signed_prediction_error[:,1]),w,mode="same") / (np.convolve(np.abs(signed_nominal_prediction_error[:,1]),w,mode="same") + 1e-3)
            residual_acc_error_ratio = np.clip(residual_acc_error_ratio,0.0,1.0)
            residual_steer_error_ratio = np.clip(residual_steer_error_ratio,0.0,1.0)
            fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(24,15), tight_layout=True)
            fig.suptitle("acc steer prediction error and std")
            axes[0,0].plot(signed_prediction_error[:,0],label="trained")
            axes[0,0].plot(signed_nominal_prediction_error[:,0],label="nominal")
            axes[0,0].set_title("acc error")
            axes[0,0].legend()
            axes[0,1].plot(std_prediction[:,0],label="acc")
            axes[0,1].set_title("acc std")
            axes[0,1].legend()
            axes[1,0].plot(signed_prediction_error[:,1],label="trained")
            axes[1,0].plot(signed_nominal_prediction_error[:,1],label="nominal")
            axes[1,0].set_title("steer error")
            axes[1,0].legend()
            axes[1,1].plot(std_prediction[:,1],label="steer")
            axes[1,1].set_title("steer std")
            axes[1,1].legend()
            axes[0,2].plot(prob_acc,label="acc_prob")
            axes[0,2].set_title("acc prob")
            axes[0,2].legend()
            axes[0,3].plot(prob_acc*prob_steer,label="acc*steer")
            axes[0,3].set_title("acc*steer prob")
            axes[0,3].legend()
            axes[1,2].plot(prob_steer,label="steer_prob")
            axes[1,2].set_title("steer prob")
            axes[1,2].legend()
            axes[1,3].plot(prob_acc_steer,label="acc_steer")
            axes[1,3].set_title("acc_steer prob")
            plt.show()
            #plt.plot(np.abs(signed_prediction_error)[:,0]/np.abs(signed_prediction_error[:,0]).max(),label="abs_trained_error/" + str(np.abs(signed_prediction_error[:,0]).max()))
            plt.plot(residual_acc_error_ratio,label="residual_acc_error_ratio")
            plt.plot(std_prediction[:,0]/std_prediction[:,0].max(),label="acc_std/"+str(std_prediction[:,0].max()))
            plt.plot(prob_acc.min()/prob_acc,label=str(prob_acc.min()) + "/acc_prob")
            plt.title("acc comparison")
            plt.legend()
            plt.show()
            plt.plot(X_test_extracted[:,1],label="acc")
            plt.title("acc")
            plt.legend()
            plt.show()
            plt.plot(X_test_extracted[:,3],label="acc_change")
            plt.title("acc_change")
            plt.legend()
            plt.show()
            #plt.plot(np.abs(signed_prediction_error)[:,1]/np.abs(signed_prediction_error[:,1]).max(),label="abs_trained_error/" + str(np.abs(signed_prediction_error[:,1]).max()))
            plt.plot(residual_steer_error_ratio,label="residual_steer_error_ratio")
            plt.plot(std_prediction[:,1]/std_prediction[:,1].max(),label="steer_std/"+str(std_prediction[:,1].max()))
            plt.plot(prob_steer.min()/prob_steer,label=str(prob_steer.min())+"/steer_prob")
            plt.title("steer comparison")
            plt.legend()
            plt.show()
            plt.plot(X_test_extracted[:,2],label="steer")
            plt.title("steer")
            plt.legend()
            plt.show()
            plt.plot(X_test_extracted[:,4],label="steer_change")
            plt.title("steer_change")
            plt.legend()
            plt.show()
        else:
            fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(24,15), tight_layout=True)
            fig.suptitle("acc steer prediction error and std")
            axes[0,0].plot(signed_prediction_error[:,0],label="trained")
            axes[0,0].plot(signed_nominal_prediction_error[:,0],label="nominal")
            axes[0,0].set_title("acc error")
            axes[0,0].legend()
            axes[0,1].plot(std_prediction[:,0],label="acc")
            axes[0,1].set_title("acc std")
            axes[0,1].legend()
            axes[1,0].plot(signed_prediction_error[:,1],label="trained")
            axes[1,0].plot(signed_nominal_prediction_error[:,1],label="nominal")
            axes[1,0].set_title("steer error")
            axes[1,0].legend()
            axes[1,1].plot(std_prediction[:,1],label="steer")
            axes[1,1].set_title("steer std")
            axes[1,1].legend()
        return result_dict
        

    def get_acc_steer_prediction(self,model,X,Y,window_size=10,batch_size=10000):
        model.eval()
        model.to("cpu")
        num_batches = (X.size(0) + batch_size - 1) // batch_size
        prediction = []
        nominal_prediction = []
        for k in range(num_batches):
            _, hc = model(X[k*batch_size:(k+1)*batch_size], previous_error=Y[k*batch_size:(k+1)*batch_size, :past_length, -2:], mode="get_lstm_states")
            states_tmp=X[k*batch_size:(k+1)*batch_size, past_length, [vel_index, acc_index, steer_index]].unsqueeze(1)
            nominal_states_tmp = states_tmp.clone()
            for i in range(window_size):
                states_tmp_with_input_history = torch.cat((states_tmp, X[k*batch_size:(k+1)*batch_size,[past_length+i], 3:]), dim=2)
                predicted_error, hc =  model(states_tmp_with_input_history, hc=hc, mode="predict_with_hc")
                states_tmp[:,0,0] = X[k*batch_size:(k+1)*batch_size,past_length + i + 1, 0]
                nominal_states_tmp[:,0,0] = X[k*batch_size:(k+1)*batch_size,past_length + i + 1, 0]
                for j in range(prediction_step):
                    states_tmp[:,0,1] = states_tmp[:,0,1] + (X[k*batch_size:(k+1)*batch_size,past_length + i, acc_input_indices_nom[j]] - states_tmp[:,0,1]) * (1 - np.exp(- control_dt / acc_time_constant))
                    states_tmp[:,0,2] = states_tmp[:,0,2] + (X[k*batch_size:(k+1)*batch_size,past_length + i, steer_input_indices_nom[j]] - states_tmp[:,0,2]) * (1 - np.exp(- control_dt / steer_time_constant))
                    nominal_states_tmp[:,0,1] = nominal_states_tmp[:,0,1] + (X[k*batch_size:(k+1)*batch_size,past_length + i, acc_input_indices_nom[j]] - nominal_states_tmp[:,0,1]) * (1 - np.exp(- control_dt / acc_time_constant))
                    nominal_states_tmp[:,0,2] = nominal_states_tmp[:,0,2] + (X[k*batch_size:(k+1)*batch_size,past_length + i, steer_input_indices_nom[j]] - nominal_states_tmp[:,0,2]) * (1 - np.exp(- control_dt / steer_time_constant))
                states_tmp[:,0,1] = states_tmp[:,0,1] + predicted_error[:,0,state_name_to_predicted_index["acc"]] * control_dt * prediction_step
                states_tmp[:,0,2] = states_tmp[:,0,2] + predicted_error[:,0,state_name_to_predicted_index["steer"]] * control_dt * prediction_step
                

            prediction.append(states_tmp[:,0,[1,2]].detach().numpy())
            nominal_prediction.append(nominal_states_tmp[:,0,[1,2]].detach().numpy())
        prediction = np.concatenate(prediction,axis=0)
        nominal_prediction = np.concatenate(nominal_prediction,axis=0)
        return prediction, nominal_prediction
    def calc_dataloader_weights(self, window_size=10, vel_scale=0.1,acc_scale=1.0,steer_scale=1.5,acc_change_scale=5.0,steer_change_scale=5.0,bandwidth=0.3, maximum_weight_by_acc_density=3.0,maximum_weight_by_steer_density=3.0, maximum_weight_by_small_steer=3.0, maximum_weight_by_small_steer_change=10.0,small_steer_threshold=0.01,small_steer_change_threshold=0.01,steer_bins=[0.01,0.1]):
        if len(self.X_train_list) == 0:
            print("no data")
            return
        if len(self.X_val_list) == 0:
            print("no validation data")
            return
        self.extract_features_for_trining_data(window_size, vel_scale,acc_scale,steer_scale,acc_change_scale,steer_change_scale,bandwidth)
        weights = []
        acc_density = np.exp(self.kde_acc.score_samples(self.X_extracted_scaled[:,[0,1,3]]))
        steer_density = np.exp(self.kde_steer.score_samples(self.X_extracted_scaled[:,[0,2,4]]))
        for i in range(self.X_extracted_scaled.shape[0]):
            weight_by_acc_density = 1.0 / (acc_density[i] + 1.0/maximum_weight_by_acc_density)
            weight_by_steer_density = 1.0 / (steer_density[i] + 1.0/maximum_weight_by_steer_density)
            weight_by_small_steer = (1.0 + 1.0/maximum_weight_by_small_steer) / (np.abs(self.X_extracted[i,4]) / small_steer_threshold + 1/maximum_weight_by_small_steer)
            #weight_by_small_steer_change = (1.0 + 1.0/maximum_weight_by_small_steer_change) / (np.abs(self.X_extracted[i,3]) / small_steer_change_threshold + 1/maximum_weight_by_small_steer_change)
            #weight_by_small_steer_and_change = max(min(weight_by_small_steer,weight_by_small_steer_change),1.0)
            weights.append(weight_by_acc_density * weight_by_steer_density * weight_by_small_steer)# * weight_by_small_steer_and_change)
        self.weights_for_dataloader = np.array(weights)
        return
        steer_positive_indices = []
        steer_negative_indices = []
        steer_positive_indices.append(np.where((self.X_extracted[:,2] > 0.0) & (self.X_extracted[:,2] < steer_bins[0]))[0])
        steer_negative_indices.append(np.where((self.X_extracted[:,2] < 0.0) & (self.X_extracted[:,2] > -steer_bins[0]))[0])
        for i in range(len(steer_bins)-1):
            steer_positive_indices.append(np.where((self.X_extracted[:,2] > steer_bins[i]) & (self.X_extracted[:,2] < steer_bins[i+1]))[0])
            steer_negative_indices.append(np.where((self.X_extracted[:,2] < -steer_bins[i]) & (self.X_extracted[:,2] > -steer_bins[i+1]))[0])
        steer_positive_indices.append(np.where(self.X_extracted[:,2] > steer_bins[-1])[0])
        steer_negative_indices.append(np.where(self.X_extracted[:,2] < -steer_bins[-1])[0])
        self.weights_for_dataloader = np.array(weights)
        for i in range(len(steer_positive_indices)):
            positive_ind = steer_positive_indices[i]
            negative_ind = steer_negative_indices[i]
            if len(positive_ind) == 0 or len(negative_ind) == 0:
                continue
            positive_weight_sum = np.sum(self.weights_for_dataloader[positive_ind])
            negative_weight_sum = np.sum(self.weights_for_dataloader[negative_ind])
            positive_weight_coef = 0.5  + 0.5 * (negative_weight_sum + 1e+3) / (positive_weight_sum + 1e-3)
            negative_weight_coef = 0.5  + 0.5 * (positive_weight_sum + 1e+3) / (negative_weight_sum + 1e-3)
            self.weights_for_dataloader[positive_ind] *= positive_weight_coef
            self.weights_for_dataloader[negative_ind] *= negative_weight_coef

    """
    def get_linear_regression_matrices(self, save_dir, batch_size=10000,mode="single_model"):
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
        X_train = torch.tensor(X_train_np, dtype=torch.float32, device=self.device)
        Y_train = torch.tensor(Y_train_np, dtype=torch.float32, device=self.device)
        num_batches = (X_train.size(0) + batch_size - 1) // batch_size
        x_dim = 3
        feature_size = 1 + x_dim * parameters.x_history_len_for_linear_compensation + acc_queue_size + steer_queue_size + 2*prediction_step
        if mode == "single_model":
            h_dim = self.model.lstm_hidden_size
        elif mode == "ensemble":
            h_dim = self.models[0].lstm_hidden_size
        else:
            print("mode is not correct")
            return
        feature_size += 2 * h_dim
        if parameters.fit_yaw_for_linear_compensation:
            target_size = 3
        else:
            target_size = 2
        XXT = np.zeros((feature_size,feature_size))
        YXT = np.zeros((target_size,feature_size))
        with torch.no_grad():
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, X_train.size(0))
                if mode == "single_model":
                    self.model.to(self.device)
                    _, hc = self.model(X_train[start_idx:end_idx], previous_error=Y_train[start_idx:end_idx, :past_length, -2:], mode="get_lstm_states")
                elif mode == "ensemble":
                    for i in range(len(self.models)):
                        self.models[i].to(self.device)
                    _, hc = self.models[0](X_train[start_idx:end_idx], previous_error=Y_train[start_idx:end_idx, :past_length, -2:], mode="get_lstm_states")
                for j in range(parameters.compensation_lstm_len):
                    x_input = np.zeros((end_idx - start_idx, feature_size))
                    x_input[:,0] = 1.0
                    x_input[:,-2*h_dim:-h_dim] = hc[0][0].to("cpu").detach().numpy()
                    x_input[:,-h_dim:] = hc[1][0].to("cpu").detach().numpy()
                    x_input[:, 1 + x_dim * parameters.x_history_len_for_linear_compensation:- 2 * h_dim] = (
                        X_train_np[start_idx:end_idx,past_length + j, 3:]
                    )
                    for k in range(parameters.x_history_len_for_linear_compensation):
                        x_input[:,1 + x_dim * k] = parameters.vel_scale_for_linear_compensation * X_train_np[start_idx:end_idx,past_length + j - parameters.x_history_len_for_linear_compensation + k, 0]
                        x_input[:,2 + x_dim * k] = X_train_np[start_idx:end_idx,past_length + j - parameters.x_history_len_for_linear_compensation + k, 1]
                        x_input[:,3 + x_dim * k] = X_train_np[start_idx:end_idx,past_length + j - parameters.x_history_len_for_linear_compensation + k, 2]
                    XXT += (x_input.T @ x_input) / ((end_idx - start_idx) * parameters.compensation_lstm_len)
                    if mode == "single_model":
                        Y_pred, hc = self.model(X_train[start_idx:end_idx, [past_length + j]], hc=hc, mode="predict_with_hc")
                    elif mode == "ensemble":
                        Y_pred, hc_update = self.models[0](X_train[start_idx:end_idx, [past_length + j]], hc=hc, mode="predict_with_hc")
                        for k in range(1,len(self.models)):
                            Y_pred_tmp, _ = self.models[k](X_train[start_idx:end_idx, [past_length + j]], hc=hc, mode="predict_with_hc")
                            Y_pred += Y_pred_tmp
                        Y_pred /= len(self.models)
                        hc = hc_update
                    prediction_error = Y_train[start_idx:end_idx, past_length + j] - Y_pred.squeeze()
                    prediction_error_np = prediction_error.to("cpu").detach().numpy()
                    if parameters.fit_yaw_for_linear_compensation:
                        YXT += prediction_error_np[:,-3:].T @ x_input / ((end_idx - start_idx) * parameters.compensation_lstm_len)
                    else:
                        YXT += prediction_error_np[:,-2:].T @ x_input / ((end_idx - start_idx) * parameters.compensation_lstm_len)
        XXT = XXT/num_batches
        YXT = YXT/num_batches
        np.savetxt(save_dir + "/XXT.csv",XXT,delimiter=",")
        np.savetxt(save_dir + "/YXT.csv",YXT,delimiter=",")

    """
    def get_linear_regression_matrices(self,save_dir, batch_size=10000,mode="single_model"):
        speed_threshold = 6.0
        steer_threshold = 0.05
        acc_threshold = 0.1
        lasso_alpha=1e-4
        max_cumulative_error = 0.1
        max_projection_dim = 10
        X_train_np, Y_train_np, _ = self.get_sequence_data(self.X_train_list, self.Y_train_list, self.Z_train_list,self.division_indices_train)
        X_train = torch.tensor(X_train_np, dtype=torch.float32, device=self.device)
        Y_train = torch.tensor(Y_train_np, dtype=torch.float32, device=self.device)
        num_batches = (X_train.size(0) + batch_size - 1) // batch_size
        x_dim = 3
        feature_size = 1 + x_dim * parameters.x_history_len_for_linear_compensation + self.acc_queue_size + self.steer_queue_size + 2*self.prediction_step
        if mode == "single_model":
            h_dim = self.model.lstm_hidden_size
        elif mode == "ensemble":
            h_dim = self.models[0].lstm_hidden_size
        else:
            print("mode is not correct")
            return
        feature_size += 2 * h_dim
        if parameters.fit_yaw_for_linear_compensation:
            target_size = 3
        else:
            target_size = 2
        
        speed_names = ["high_speed","low_speed"]
        steer_names = ["left","right","straight"]
        acc_names = ["accelerate","decelerate","constant"]
        X_dict = {}
        Y_dict = {}
        linear_models = {}
        for speed_name in speed_names:
            for steer_name in steer_names:
                for acc_name in acc_names:
                    X_dict[speed_name + "_" + steer_name + "_" + acc_name] = []
                    Y_dict[speed_name + "_" + steer_name + "_" + acc_name] = []
                    linear_models[speed_name + "_" + steer_name + "_" + acc_name] = Lasso(alpha=lasso_alpha)
        X_train_np = X_train.to("cpu").detach().numpy()

        with torch.no_grad():
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, X_train.size(0))
                if mode == "single_model":
                    self.model.to(self.device)
                    _, hc = self.model(X_train[start_idx:end_idx], previous_error=Y_train[start_idx:end_idx, :self.past_length, -2:], mode="get_lstm_states")
                elif mode == "ensemble":
                    for i in range(len(self.models)):
                        self.models[i].to(self.device)
                    _, hc = self.models[0](X_train[start_idx:end_idx], previous_error=Y_train[start_idx:end_idx, :self.past_length, -2:], mode="get_lstm_states")
                for j in range(parameters.compensation_lstm_len):
                    x_input = np.zeros((end_idx - start_idx, feature_size-1))
                    x_input[:,-2*h_dim:-h_dim] = hc[0][0].to("cpu").detach().numpy()
                    x_input[:,-h_dim:] = hc[1][0].to("cpu").detach().numpy()
                    x_input[:, x_dim * parameters.x_history_len_for_linear_compensation:- 2 * h_dim] = (
                        X_train_np[start_idx:end_idx,self.past_length + j, 3:]
                    )
                    for k in range(parameters.x_history_len_for_linear_compensation):
                        x_input[:,x_dim * k] = parameters.vel_scale_for_linear_compensation * X_train_np[start_idx:end_idx,self.past_length + j - parameters.x_history_len_for_linear_compensation + k, 0]
                        x_input[:,x_dim * k] = X_train_np[start_idx:end_idx,self.past_length + j - parameters.x_history_len_for_linear_compensation + k, 1]
                        x_input[:,x_dim * k] = X_train_np[start_idx:end_idx,self.past_length + j - parameters.x_history_len_for_linear_compensation + k, 2]
                    if mode == "single_model":
                        Y_pred, hc = self.model(X_train[start_idx:end_idx, [self.past_length + j]], hc=hc, mode="predict_with_hc")
                    elif mode == "ensemble":
                        Y_pred, hc_update = self.models[0](X_train[start_idx:end_idx, [self.past_length + j]], hc=hc, mode="predict_with_hc")
                        for k in range(1,len(self.models)):
                            Y_pred_tmp, _ = self.models[k](X_train[start_idx:end_idx, [self.past_length + j]], hc=hc, mode="predict_with_hc")
                            Y_pred += Y_pred_tmp
                        Y_pred /= len(self.models)
                        hc = hc_update
                    prediction_error = Y_train[start_idx:end_idx, self.past_length + j] - Y_pred.squeeze()
                    prediction_error_np = prediction_error.to("cpu").detach().numpy()
                    if parameters.fit_yaw_for_linear_compensation:
                        prediction_error_np = prediction_error_np[:, -3:]
                    else:
                        prediction_error_np = prediction_error_np[:, -2:]
                    for k in range(end_idx - start_idx):
                        speed_idx = 0 if X_train_np[start_idx + k, self.past_length + j, 0] > speed_threshold else 1
                        steer_idx = 0 if X_train_np[start_idx + k, self.past_length + j, 1] < -steer_threshold else 1 if X_train_np[start_idx + k, self.past_length + j, 1] > steer_threshold else 2
                        acc_idx = 0 if X_train_np[start_idx + k, self.past_length + j, 2] > acc_threshold else 1 if X_train_np[start_idx + k, self.past_length + j, 2] < -acc_threshold else 2
                        X_dict[speed_names[speed_idx] + "_" + steer_names[steer_idx] + "_" + acc_names[acc_idx]].append(x_input[k])
                        Y_dict[speed_names[speed_idx] + "_" + steer_names[steer_idx] + "_" + acc_names[acc_idx]].append(prediction_error_np[k])
        coef_matrix_stacked = []
        for speed_name in speed_names:
            for steer_name in steer_names:
                for acc_name in acc_names:
                    X_dict[speed_name + "_" + steer_name + "_" + acc_name] = np.array(X_dict[speed_name + "_" + steer_name + "_" + acc_name])
                    Y_dict[speed_name + "_" + steer_name + "_" + acc_name] = np.array(Y_dict[speed_name + "_" + steer_name + "_" + acc_name])
                    error_std = np.std(Y_dict[speed_name + "_" + steer_name + "_" + acc_name], axis=0)
                    linear_models[speed_name + "_" + steer_name + "_" + acc_name].fit(X_dict[speed_name + "_" + steer_name + "_" + acc_name], Y_dict[speed_name + "_" + steer_name + "_" + acc_name])
                    coef_matrix_stacked.append((linear_models[speed_name + "_" + steer_name + "_" + acc_name].coef_.T / (error_std + 1e-9)).T)
        coef_matrix_stacked = np.vstack(coef_matrix_stacked)
        U, S, VT = np.linalg.svd(coef_matrix_stacked)

        cumulative_sum = np.cumsum(S)
        cumulative_ratio = cumulative_sum / cumulative_sum[-1]
        print("cumulative_ratio:", cumulative_ratio)
        projection_dim = max(2,min(np.where(cumulative_ratio > 1 - max_cumulative_error)[0][0] + 1, max_projection_dim))
        P = VT[:projection_dim]
        np.savetxt(save_dir + "/Projection.csv", P, delimiter=",")
    def update_adaptive_weight(self,model, X, Y, batch_size=10000):
        if model is not None:
            model.to(self.device)
            model.eval()
        num_batches = (X.size(0) + batch_size - 1) // batch_size
        prediction_error = torch.zeros(self.adaptive_weight.shape[0], device=self.device)
        with torch.no_grad():
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, X.size(0))
                
                X_batch = X[start_idx:end_idx]
                Y_batch = Y[start_idx:end_idx]
                if model is not None:
                    Y_pred, _ = model(X_batch, previous_error=Y_batch[:, :past_length, -2:], mode="get_lstm_states")
                    prediction_error += torch.mean(torch.abs(Y_pred - Y_batch[:,past_length:]),dim=(0,1)) * (end_idx - start_idx)
                else:
                    prediction_error += torch.mean(torch.abs(Y_batch[:,past_length:]),dim=(0,1)) * (end_idx - start_idx)
            prediction_error /= X.size(0)
        self.adaptive_weight = 1.0 / (prediction_error + 1e-4)
        for i in range(len(self.adaptive_weight)):
            if self.adaptive_weight[i] > torch.max(self.adaptive_weight[-2:]):
                self.adaptive_weight[i] = torch.max(self.adaptive_weight[-2:]) # acc and steer are respected
        self.adaptive_weight = self.adaptive_weight / torch.mean(self.adaptive_weight)
