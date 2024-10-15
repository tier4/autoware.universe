import numpy as np
import json
import torch
import os
from pathlib import Path
import csv
package_path_json = str(Path(__file__).parent.parent) + "/package_path.json"
with open(package_path_json, "r") as file:
    package_path = json.load(file)
default_path = (
    package_path["path"] + "/vehicle_models"
)
with open(package_path_json, "r") as file:
    package_path = json.load(file)
def initialize_vehicle_adaptor(vehicle_adaptor,vehicle_adaptor_model_path=default_path):
    i = 0
    vehicle_adaptor.clear_NN_params()
    while os.path.exists(vehicle_adaptor_model_path + "/vehicle_model_" + str(i) + ".pth"):


        state_component_predicted_dir = vehicle_adaptor_model_path + "/vehicle_model_" + str(i) + "/state_component_predicted.csv"
        state_component_predicted = []
        with open(state_component_predicted_dir, "r", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                state_component_predicted.extend(row)
        vehicle_model = torch.load(vehicle_adaptor_model_path + "/vehicle_model_" + str(i) + ".pth")
        weight_encoder_ih = []
        weight_encoder_hh = []
        bias_encoder_ih = []
        bias_encoder_hh = []
        for i in range(vehicle_model.num_layers_encoder):
            weight_encoder_ih.append(vehicle_model.lstm_encoder.__getattr__("weight_ih_l" + str(i)).detach().numpy().astype(np.float64))
            weight_encoder_hh.append(vehicle_model.lstm_encoder.__getattr__("weight_hh_l" + str(i)).detach().numpy().astype(np.float64))
            bias_encoder_ih.append(vehicle_model.lstm_encoder.__getattr__("bias_ih_l" + str(i)).detach().numpy().astype(np.float64))
            bias_encoder_hh.append(vehicle_model.lstm_encoder.__getattr__("bias_hh_l" + str(i)).detach().numpy().astype(np.float64))
        vehicle_adaptor.set_NN_params(
            vehicle_model.acc_encoder_layer_1[0].weight.detach().numpy().astype(np.float64),
            vehicle_model.steer_encoder_layer_1[0].weight.detach().numpy().astype(np.float64),
            vehicle_model.acc_encoder_layer_2[0].weight.detach().numpy().astype(np.float64),
            vehicle_model.steer_encoder_layer_2[0].weight.detach().numpy().astype(np.float64),
            vehicle_model.acc_layer_1[0].weight.detach().numpy().astype(np.float64),
            vehicle_model.steer_layer_1[0].weight.detach().numpy().astype(np.float64),
            vehicle_model.acc_layer_2[0].weight.detach().numpy().astype(np.float64),
            vehicle_model.steer_layer_2[0].weight.detach().numpy().astype(np.float64),
            weight_encoder_ih,
            weight_encoder_hh,
            vehicle_model.lstm.weight_ih_l0.detach().numpy().astype(np.float64),
            vehicle_model.lstm.weight_hh_l0.detach().numpy().astype(np.float64),
            vehicle_model.complimentary_layer[0].weight.detach().numpy().astype(np.float64),
            vehicle_model.linear_relu[0].weight.detach().numpy().astype(np.float64),
            vehicle_model.final_layer.weight.detach().numpy().astype(np.float64),
            vehicle_model.acc_encoder_layer_1[0].bias.detach().numpy().astype(np.float64),
            vehicle_model.steer_encoder_layer_1[0].bias.detach().numpy().astype(np.float64),
            vehicle_model.acc_encoder_layer_2[0].bias.detach().numpy().astype(np.float64),
            vehicle_model.steer_encoder_layer_2[0].bias.detach().numpy().astype(np.float64),
            vehicle_model.acc_layer_1[0].bias.detach().numpy().astype(np.float64),
            vehicle_model.steer_layer_1[0].bias.detach().numpy().astype(np.float64),
            vehicle_model.acc_layer_2[0].bias.detach().numpy().astype(np.float64),
            vehicle_model.steer_layer_2[0].bias.detach().numpy().astype(np.float64),
            bias_encoder_ih,
            bias_encoder_hh,
            vehicle_model.lstm.bias_ih_l0.detach().numpy().astype(np.float64),
            vehicle_model.lstm.bias_hh_l0.detach().numpy().astype(np.float64),
            vehicle_model.complimentary_layer[0].bias.detach().numpy().astype(np.float64),
            vehicle_model.linear_relu[0].bias.detach().numpy().astype(np.float64),
            vehicle_model.final_layer.bias.detach().numpy().astype(np.float64),
            vehicle_model.vel_scaling,
            vehicle_model.vel_bias,
            state_component_predicted,
        )
        i += 1

    vehicle_adaptor.send_initialized_flag()
