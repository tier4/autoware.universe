import numpy as np
import json
import torch
import os
from pathlib import Path

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
        vehicle_model = torch.load(vehicle_adaptor_model_path + "/vehicle_model_" + str(i) + ".pth")
        vehicle_adaptor.set_NN_params(
            vehicle_model.acc_layer_1[0].weight.detach().numpy().astype(np.float64),
            vehicle_model.steer_layer_1[0].weight.detach().numpy().astype(np.float64),
            vehicle_model.acc_layer_2[0].weight.detach().numpy().astype(np.float64),
            vehicle_model.steer_layer_2[0].weight.detach().numpy().astype(np.float64),
            vehicle_model.lstm.weight_ih_l0.detach().numpy().astype(np.float64),
            vehicle_model.lstm.weight_hh_l0.detach().numpy().astype(np.float64),
            vehicle_model.complimentary_layer[0].weight.detach().numpy().astype(np.float64),
            vehicle_model.linear_relu[0].weight.detach().numpy().astype(np.float64),
            vehicle_model.final_layer.weight.detach().numpy().astype(np.float64),
            vehicle_model.acc_layer_1[0].bias.detach().numpy().astype(np.float64),
            vehicle_model.steer_layer_1[0].bias.detach().numpy().astype(np.float64),
            vehicle_model.acc_layer_2[0].bias.detach().numpy().astype(np.float64),
            vehicle_model.steer_layer_2[0].bias.detach().numpy().astype(np.float64),
            vehicle_model.lstm.bias_hh_l0.detach().numpy().astype(np.float64),
            vehicle_model.lstm.bias_ih_l0.detach().numpy().astype(np.float64),
            vehicle_model.complimentary_layer[0].bias.detach().numpy().astype(np.float64),
            vehicle_model.linear_relu[0].bias.detach().numpy().astype(np.float64),
            vehicle_model.final_layer.bias.detach().numpy().astype(np.float64),
            vehicle_model.vel_scaling,
            vehicle_model.vel_bias,
        )
        i += 1

    vehicle_adaptor.send_initialized_flag()
