import sys

from utils.data_collection_utils import ControlType
from autoware_vehicle_adaptor.training import train_error_prediction_NN
import numpy as np
import python_simulator
import simplejson as json
import os

#states_ref_mode = "predict_by_polynomial_regression"
#states_ref_mode = "controller_prediction"
states_ref_mode = "controller_d_inputs_schedule"
simulator = python_simulator.PythonSimulator()
if not os.path.isdir("log_data"):
    os.mkdir("log_data")
if os.path.isfile("supporting_data/sim_setting.json"):
    with open("supporting_data/sim_setting.json", "r") as file:
        sim_setting_dict = json.load(file)
        print("load sim_setting.json")
        print("sim_setting_dict", sim_setting_dict)
        simulator.perturbed_sim(sim_setting_dict)   
else:
    print("sim_setting.json not found")
if len(sys.argv) > 1:
    if sys.argv[1] == "nominal_test":
        save_dir = "log_data/test_python_nominal_sim"
        simulator.drive_sim(save_dir=save_dir)
    else:
        print("Invalid argument")
        sys.exit(1)
else:
    train_dir = "log_data/test_python_pure_pursuit_train"
    val_dir = "log_data/test_python_pure_pursuit_val"
    simulator.drive_sim(
        data_collection_seed=0,
        max_control_time = 900.0,
        control_type=ControlType.pp_eight,
        save_dir=train_dir,
    )
    simulator.drive_sim(
        data_collection_seed=1,
        max_control_time = 900.0,
        control_type=ControlType.pp_eight,
        save_dir=val_dir,
    )
    model_trainer = train_error_prediction_NN.train_error_prediction_NN()
    model_trainer.add_data_from_csv(train_dir, add_mode="as_train")
    model_trainer.add_data_from_csv(val_dir, add_mode="as_val")
    model_trainer.get_trained_ensemble_models(batch_size=500,ensemble_size=5)
    paths = [train_dir + "/vehicle_model_1.pth", train_dir + "/vehicle_model_2.pth", train_dir + "/vehicle_model_3.pth", train_dir + "/vehicle_model_4.pth", train_dir + "/vehicle_model_5.pth"]
    model_trainer.save_ensemble_models(paths=paths)
    save_dir = "log_data/test_python_vehicle_adaptor_sim"
    simulator.drive_sim(
        save_dir=save_dir, use_vehicle_adaptor=True, vehicle_adaptor_model_path=paths, states_ref_mode=states_ref_mode
    )
