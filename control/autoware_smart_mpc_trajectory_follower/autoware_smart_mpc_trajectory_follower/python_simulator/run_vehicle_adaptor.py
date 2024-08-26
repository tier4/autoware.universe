import sys

from assets import ControlType
from autoware_vehicle_adaptor.training import train_error_prediction_NN
import numpy as np
import python_simulator

initial_error = np.array(
    [0.001, 0.03, 0.01, -0.001, 0, 2 * python_simulator.measurement_steer_bias]
)

#states_ref_mode = "predict_by_polynomial_regression"
states_ref_mode = "controller_prediction"
#states_ref_mode = "controller_d_input_schedule"
if len(sys.argv) > 1:
    if sys.argv[1] == "nominal_test":
        save_dir = "test_python_nominal_sim"
        python_simulator.drive_sim(save_dir=save_dir, initial_error=initial_error)
    else:
        print("Invalid argument")
        sys.exit(1)
else:
    train_dir = "test_python_pure_pursuit_train"
    val_dir = "test_python_pure_pursuit_val"
    python_simulator.drive_sim(
        seed=0,
        t_range=[0, 900.0],
        control_type=ControlType.pp_eight,
        save_dir=train_dir,
        initial_error=initial_error,
    )
    python_simulator.drive_sim(
        seed=1,
        t_range=[0, 900.0],
        control_type=ControlType.pp_eight,
        save_dir=val_dir,
        initial_error=initial_error,
    )
    model_trainer = train_error_prediction_NN.train_error_prediction_NN()
    model_trainer.add_data_from_csv(train_dir, add_mode="as_train")
    model_trainer.add_data_from_csv(val_dir, add_mode="as_val")
    model_trainer.get_trained_model(batch_size=100)
    model_trainer.save_model(path=train_dir+"/vehicle_model.pth")

    save_dir = "test_python_vehicle_adaptor_sim"
    python_simulator.drive_sim(
        save_dir=save_dir, initial_error=initial_error, use_vehicle_adaptor=True, vehicle_adaptor_model_path=train_dir+"/vehicle_model.pth", states_ref_mode=states_ref_mode
    )
