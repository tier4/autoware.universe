import numpy as np
import torch
from torch import nn

acc_queue_size = 15
steer_queue_size = 15
prediction_step = 3

class ErrorPredictionNN(nn.Module):
    def __init__(
        self,
        states_size,
        vel_index,
        output_size,
        prediction_length,
        acc_hidden_sizes=(16, 8),
        steer_hidden_sizes=(16, 8),
        lstm_hidden_size=16,
        complimentary_size=64,
        hidden_size=16,
        randomize=0.01,
        vel_scaling=1.0,
        vel_bias=0.0,
    ):
        super(ErrorPredictionNN, self).__init__()
        acc_index = states_size - 2
        steer_index = states_size - 1
        self.states_size = states_size
        self.vel_index = vel_index
        self.vel_scaling = vel_scaling
        self.vel_bias = vel_bias
        acc_input_indices = np.arange(states_size, states_size + acc_queue_size + prediction_step)
        steer_input_indices = np.arange(
            states_size + acc_queue_size + prediction_step, states_size + acc_queue_size + steer_queue_size + 2 * prediction_step
        )
        self.acc_layer_input_indices = np.concatenate(([vel_index, acc_index], acc_input_indices))
        self.steer_layer_input_indices = np.concatenate(
            ([vel_index, steer_index], steer_input_indices)
        )
        self.prediction_length = prediction_length
        lb = -randomize
        ub = randomize
        self.acc_layer_1 = nn.Sequential(
            nn.Linear(len(self.acc_layer_input_indices), acc_hidden_sizes[0]), nn.ReLU()
        )
        nn.init.uniform_(self.acc_layer_1[0].weight, a=lb, b=ub)
        nn.init.uniform_(self.acc_layer_1[0].bias, a=lb, b=ub)
        self.steer_layer_1 = nn.Sequential(
            nn.Linear(len(self.steer_layer_input_indices), steer_hidden_sizes[0]), nn.ReLU()
        )
        nn.init.uniform_(self.steer_layer_1[0].weight, a=lb, b=ub)
        nn.init.uniform_(self.steer_layer_1[0].bias, a=lb, b=ub)
        self.acc_layer_2 = nn.Sequential(
            nn.Linear(acc_hidden_sizes[0], acc_hidden_sizes[1]), nn.ReLU()
        )
        nn.init.uniform_(self.acc_layer_2[0].weight, a=lb, b=ub)
        nn.init.uniform_(self.acc_layer_2[0].bias, a=lb, b=ub)
        self.steer_layer_2 = nn.Sequential(
            nn.Linear(steer_hidden_sizes[0], steer_hidden_sizes[1]), nn.ReLU()
        )
        nn.init.uniform_(self.steer_layer_2[0].weight, a=lb, b=ub)
        nn.init.uniform_(self.steer_layer_2[0].bias, a=lb, b=ub)
        combined_input_size = states_size - 2 + acc_hidden_sizes[1] + steer_hidden_sizes[1]
        self.lstm = nn.LSTM(combined_input_size, lstm_hidden_size, batch_first=True)
        nn.init.uniform_(self.lstm.weight_hh_l0, a=lb, b=ub)
        nn.init.uniform_(self.lstm.weight_ih_l0, a=lb, b=ub)
        nn.init.uniform_(self.lstm.bias_hh_l0, a=lb, b=ub)
        nn.init.uniform_(self.lstm.bias_ih_l0, a=lb, b=ub)
        self.complimentary_layer = nn.Sequential(
            nn.Linear(combined_input_size, complimentary_size), nn.ReLU()
        )
        nn.init.uniform_(self.complimentary_layer[0].weight, a=lb, b=ub)
        nn.init.uniform_(self.complimentary_layer[0].bias, a=lb, b=ub)
        self.linear_relu = nn.Sequential(
            nn.Linear(lstm_hidden_size + complimentary_size, hidden_size),
            nn.ReLU(),
        )
        nn.init.uniform_(self.linear_relu[0].weight, a=lb, b=ub)
        nn.init.uniform_(self.linear_relu[0].bias, a=lb, b=ub)
        self.final_layer = nn.Linear(hidden_size, output_size)
        nn.init.uniform_(self.final_layer.weight, a=lb, b=ub)
        nn.init.uniform_(self.final_layer.bias, a=lb, b=ub)
    def forward(self, x, hc=None, mode="default"):
        x_vel_scaled = x.clone()
        x_vel_scaled[:,:,self.vel_index] = (x_vel_scaled[:,:,self.vel_index] - self.vel_bias ) * self.vel_scaling
        acc_input = x_vel_scaled[:, :, self.acc_layer_input_indices]
        steer_input = x_vel_scaled[:, :, self.steer_layer_input_indices]
        acc_output = self.acc_layer_1(acc_input)
        steer_output = self.steer_layer_1(steer_input)
        acc_output = self.acc_layer_2(acc_output)
        steer_output = self.steer_layer_2(steer_output)
        lstm_input = torch.cat((x_vel_scaled[:, :, : self.states_size - 2], acc_output, steer_output), dim=2)
        if mode == "default" or mode == "get_lstm_states":
            _, hc = self.lstm(lstm_input[:, : -self.prediction_length])
            lstm_tail, _ = self.lstm(lstm_input[:, -self.prediction_length :], hc)
            complimentary_output = self.complimentary_layer(lstm_input[:,-self.prediction_length :])
            lstm_output = torch.cat((lstm_tail, complimentary_output), dim=2)
            output = self.linear_relu(lstm_output)
            output = self.final_layer(output)
            if mode == "default":
                return output
            elif mode == "get_lstm_states":
                return output, hc
        elif mode == "predict_with_hc":
            lstm, hc_new = self.lstm(lstm_input, hc)
            complimentary_output = self.complimentary_layer(lstm_input)
            lstm_output = torch.cat((lstm, complimentary_output), dim=2)
            output = self.linear_relu(lstm_output)
            output = self.final_layer(output)
            return output, hc_new

