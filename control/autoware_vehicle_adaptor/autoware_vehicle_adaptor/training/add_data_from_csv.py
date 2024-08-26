import csv
import glob
import json
import os
from pathlib import Path

from autoware_vehicle_adaptor.scripts import utils
from autoware_vehicle_adaptor.calibrator import get_acc_input_from_csv_via_map
import numpy as np
import scipy.interpolate
from scipy.ndimage import gaussian_filter
from scipy.spatial.transform import Rotation
import os

wheel_base = 2.79
acc_queue_size = 15
steer_queue_size = 15
acc_delay_step = 3
acc_time_delay = 0.1
steer_delay_step = 8
steer_time_delay = 0.27
x_index = 0
y_index = 1
vel_index = 2
yaw_index = 3
acc_index = 4
steer_index = 5
acc_input_indices = np.arange(6, 6 + acc_queue_size)
steer_input_indices = np.arange(6 + acc_queue_size, 6 + acc_queue_size + steer_queue_size)

control_dt = 0.033
predict_step = 3
predict_dt = predict_step * control_dt

acc_time_constant = 0.1
steer_time_constant = 0.27

x_error_sigma_for_training = 30.0
y_error_sigma_for_training = 30.0
v_error_sigma_for_training = 30.0
theta_error_sigma_for_training = 10.0
acc_error_sigma_for_training = 5.0
steer_error_sigma_for_training = 5.0


def data_smoothing(data: np.ndarray, sigma: float) -> np.ndarray:
    """Apply a Gaussian filter to the data."""
    data_ = gaussian_filter(data, sigma)
    return data_


def yaw_transform(raw_yaw: np.ndarray) -> np.ndarray:
    """Adjust and transform within a period of 2π so that the yaw angle is continuous."""
    transformed_yaw = np.zeros(raw_yaw.shape)
    transformed_yaw[0] = raw_yaw[0]
    for i in range(raw_yaw.shape[0] - 1):
        rotate_num = (raw_yaw[i + 1] - transformed_yaw[i]) // (2 * np.pi)
        if raw_yaw[i + 1] - transformed_yaw[i] - 2 * rotate_num * np.pi < np.pi:
            transformed_yaw[i + 1] = raw_yaw[i + 1] - 2 * rotate_num * np.pi
        else:
            transformed_yaw[i + 1] = raw_yaw[i + 1] - 2 * (rotate_num + 1) * np.pi
    return transformed_yaw


class add_data_from_csv:
    def __init__(self):
        self.X_train_list = []
        self.Y_train_list = []
        self.Z_train_list = []
        self.X_val_list = []
        self.Y_val_list = []
        self.Z_val_list = []
        self.division_indices_train = []
        self.division_indices_val = []
        self.nominal_dynamics = utils.NominalDynamics()
        self.nominal_dynamics.set_params(
            wheel_base,
            acc_time_delay,
            steer_time_delay,
            acc_time_constant,
            steer_time_constant,
            acc_queue_size,
            steer_queue_size,
            control_dt,
            predict_step,
        )
    def clear_data(self):
        self.X_train_list = []
        self.Y_train_list = []
        self.Z_train_list = []
        self.X_val_list = []
        self.Y_val_list = []
        self.Z_val_list = []
        self.division_indices_train = []
        self.division_indices_val = []
    def add_data_from_csv(self, dir_name: str, add_mode="divide",map_dir=None) -> None:
        kinematic = np.loadtxt(
            dir_name + "/kinematic_state.csv", delimiter=",", usecols=[0, 1, 4, 5, 7, 8, 9, 10, 47]
        )
        pose_position_x = kinematic[:, 2]
        pose_position_y = kinematic[:, 3]
        vel = kinematic[:, 8]
        raw_yaw = Rotation.from_quat(kinematic[:, 4:8]).as_euler("xyz")[:, 2]
        yaw = yaw_transform(raw_yaw)

        loc_acc = np.loadtxt(dir_name + "/acceleration.csv", delimiter=",", usecols=[0, 1, 3])
        acc = loc_acc[:, 2]

        vehicle_steer = np.loadtxt(
            dir_name + "/steering_status.csv", delimiter=",", usecols=[0, 1, 2]
        )
        steer = vehicle_steer[:, 2]
        if os.path.exists(dir_name + "/vehicle_adaptor_cmd.csv"):
            control_cmd = np.loadtxt(
                dir_name + "/vehicle_adaptor_cmd.csv", delimiter=","
            )
            acc_cmd = control_cmd[:, [0,1,2]]
            steer_des = control_cmd[:,3]
        else:
            control_cmd = np.loadtxt(
                dir_name + "/control_cmd_orig.csv", delimiter=",", usecols=[0, 1, 8, 16]
            )
            acc_cmd = control_cmd[:, [0,1,3]]
            steer_des = control_cmd[:, 2]
        if map_dir is not None:
            acc_cmd = get_acc_input_from_csv_via_map.transform_accel_and_brake_to_acc_via_map(csv_dir=dir_name, map_dir=map_dir)
        operation_mode = np.loadtxt(
            dir_name + "/system_operation_mode_state.csv", delimiter=",", usecols=[0, 1, 2]
        )
        if operation_mode.ndim == 1:
            operation_mode = operation_mode.reshape(1, -1)
        with open(dir_name + "/system_operation_mode_state.csv") as f:
            reader = csv.reader(f, delimiter=",")
            autoware_control_enabled_str = np.array([row[3] for row in reader])

        control_enabled = np.zeros(operation_mode.shape[0])
        for i in range(operation_mode.shape[0]):
            if operation_mode[i, 2] > 1.5 and autoware_control_enabled_str[i] == "True":
                control_enabled[i] = 1.0
        for i in range(operation_mode.shape[0] - 1):
            if control_enabled[i] < 0.5 and control_enabled[i + 1] > 0.5:
                operation_start_time = operation_mode[i + 1, 0] + 1e-9 * operation_mode[i + 1, 1]
            elif control_enabled[i] > 0.5 and control_enabled[i + 1] < 0.5:
                operation_end_time = operation_mode[i + 1, 0] + 1e-9 * operation_mode[i + 1, 1]
                break
            operation_end_time = kinematic[-1, 0] + 1e-9 * kinematic[-1, 1]
        if operation_mode.shape[0] == 1:
            operation_end_time = kinematic[-1, 0] + 1e-9 * kinematic[-1, 1]
        if control_enabled[0] > 0.5:
            operation_start_time = operation_mode[0, 0] + 1e-9 * operation_mode[0, 1]
        print("operation_start_time", operation_start_time)
        print("operation_end_time", operation_end_time)

        min_time_stamp = max(
            [
                operation_start_time,
                kinematic[0, 0] + 1e-9 * kinematic[0, 1],
                loc_acc[0, 0] + 1e-9 * loc_acc[0, 1],
                vehicle_steer[0, 0] + 1e-9 * vehicle_steer[0, 1],
                acc_cmd[0, 0] + 1e-9 * acc_cmd[0, 1],
                control_cmd[0, 0] + 1e-9 * control_cmd[0, 1],
            ]
        )
        max_time_stamp = min(
            [
                operation_end_time,
                kinematic[-1, 0] + 1e-9 * kinematic[-1, 1],
                loc_acc[-1, 0] + 1e-9 * loc_acc[-1, 1],
                vehicle_steer[-1, 0] + 1e-9 * vehicle_steer[-1, 1],
                acc_cmd[-1, 0] + 1e-9 * acc_cmd[-1, 1],
                control_cmd[-1, 0] + 1e-9 * control_cmd[-1, 1],
            ]
        )

        trajectory_interpolator_list = []
        trajectory_interpolator_list.append(
            scipy.interpolate.interp1d(kinematic[:, 0] + 1e-9 * kinematic[:, 1], pose_position_x)
        )
        trajectory_interpolator_list.append(
            scipy.interpolate.interp1d(kinematic[:, 0] + 1e-9 * kinematic[:, 1], pose_position_y)
        )
        trajectory_interpolator_list.append(
            scipy.interpolate.interp1d(kinematic[:, 0] + 1e-9 * kinematic[:, 1], vel)
        )
        trajectory_interpolator_list.append(
            scipy.interpolate.interp1d(kinematic[:, 0] + 1e-9 * kinematic[:, 1], yaw)
        )
        trajectory_interpolator_list.append(
            scipy.interpolate.interp1d(loc_acc[:, 0] + 1e-9 * loc_acc[:, 1], acc)
        )
        trajectory_interpolator_list.append(
            scipy.interpolate.interp1d(vehicle_steer[:, 0] + 1e-9 * vehicle_steer[:, 1], steer)
        )
        trajectory_interpolator_list.append(
            scipy.interpolate.interp1d(acc_cmd[:, 0] + 1e-9 * acc_cmd[:, 1], acc_cmd[:,2])
        )
        trajectory_interpolator_list.append(
            scipy.interpolate.interp1d(control_cmd[:, 0] + 1e-9 * control_cmd[:, 1], steer_des)
        )

        def get_interpolated_state(s):
            state = np.zeros(6)
            for i in range(6):
                state[i] = trajectory_interpolator_list[i](s)
            return state

        def get_interpolated_input(s):
            input = np.zeros(2)
            for i in range(2):
                input[i] = trajectory_interpolator_list[6 + i](s)
            return input

        s = min_time_stamp
        States = []
        Inputs = []

        TimeStamps = []

        while True:
            if s > max_time_stamp:
                break
            States.append(get_interpolated_state(s))

            Inputs.append(get_interpolated_input(s))

            TimeStamps.append(s)
            s += control_dt
        States = np.array(States)
        Inputs = np.array(Inputs)
        X_list = []
        Y_list = []
        Z_list = []
        TimeStamp_list = []

        for i in range(max(acc_queue_size, steer_queue_size) + 1, States.shape[0] - predict_step - 1):
            acc_input_queue = Inputs[i - acc_queue_size : i + predict_step, 0]
            steer_input_queue = Inputs[i - steer_queue_size : i + predict_step, 1]
            state_old = States[i - 1]
            state = States[i]
            state_obs = States[i + predict_step]
            x_dot = (state[x_index] - state_old[x_index]) / control_dt
            y_dot = (state[y_index] - state_old[y_index]) / control_dt
            yaw_dot = (state[yaw_index] - state_old[yaw_index]) / control_dt
            dot_info = np.array([x_dot, y_dot, yaw_dot])
            dot_info = utils.rotate_data(dot_info, state[yaw_index])
            TimeStamp_list.append(TimeStamps[i])
            X_list.append(
                np.concatenate(
                    (
                        dot_info,
                        state[[vel_index, acc_index, steer_index]],
                        acc_input_queue,
                        steer_input_queue,
                    )
                )
            )
            u_for_predict_nom = np.zeros((predict_step, 2))
            u_for_predict_nom[:, 0] = acc_input_queue[
                acc_input_queue.shape[0] - acc_delay_step - predict_step : acc_input_queue.shape[0] - acc_delay_step
            ]
            u_for_predict_nom[:, 1] = steer_input_queue[
                steer_input_queue.shape[0] - steer_delay_step - predict_step : steer_input_queue.shape[0] - steer_delay_step
            ]
            predict_error = state_obs - self.nominal_dynamics.F_nominal_predict(
                state, u_for_predict_nom.flatten()
            )
            predict_error = utils.rotate_data(predict_error, state[yaw_index])
            Y_list.append(predict_error / predict_dt)
            Z_list.append(state)

        Y_smooth = np.array(Y_list)
        Y_smooth[:, x_index] = data_smoothing(Y_smooth[:, x_index], x_error_sigma_for_training)
        Y_smooth[:, y_index] = data_smoothing(Y_smooth[:, y_index], y_error_sigma_for_training)
        Y_smooth[:, vel_index] = data_smoothing(Y_smooth[:, vel_index], v_error_sigma_for_training)
        Y_smooth[:, yaw_index] = data_smoothing(
            Y_smooth[:, yaw_index], theta_error_sigma_for_training
        )
        Y_smooth[:, acc_index] = data_smoothing(
            Y_smooth[:, acc_index], acc_error_sigma_for_training
        )
        Y_smooth[:, steer_index] = data_smoothing(
            Y_smooth[:, steer_index], steer_error_sigma_for_training
        )

        if add_mode == "divide":
            for i in range(len(X_list)):
                if i < 3 * len(X_list) / 4:
                    self.X_train_list.append(X_list[i])
                    self.Y_train_list.append(Y_smooth[i])
                    self.Z_train_list.append(Z_list[i])
                else:
                    self.X_val_list.append(X_list[i])
                    self.Y_val_list.append(Y_smooth[i])
                    self.Z_val_list.append(Z_list[i])

            self.division_indices_train.append(len(self.X_train_list))
            self.division_indices_val.append(len(self.X_val_list))

        elif add_mode == "as_train":
            for i in range(len(X_list)):
                self.X_train_list.append(X_list[i])
                self.Y_train_list.append(Y_smooth[i])
                self.Z_train_list.append(Z_list[i])

            self.division_indices_train.append(len(self.X_train_list))
        if add_mode == "as_val":
            for i in range(len(X_list)):
                self.X_val_list.append(X_list[i])
                self.Y_val_list.append(Y_smooth[i])
                self.Z_val_list.append(Z_list[i])

            self.division_indices_val.append(len(self.X_val_list))
