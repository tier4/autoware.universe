# Copyright 2024 Proxima Technology Inc, TIER IV
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import scipy.interpolate
from autoware_vehicle_adaptor.scripts import actuation_map_2d
control_dt = 0.033
def transform_accel_and_brake_to_acc_via_map(csv_dir, map_dir):
    accel_map_control = actuation_map_2d.ActuationMap2D(map_dir + "/accel_map.csv")
    brake_map_control = actuation_map_2d.ActuationMap2D(map_dir + "/brake_map.csv")
    accel_with_timestamp = np.loadtxt(csv_dir + "/accel_cmd.csv", delimiter=",")
    brake_with_timestamp = np.loadtxt(csv_dir + "/brake_cmd.csv", delimiter=",")
    kinematic = np.loadtxt(
        csv_dir + "/kinematic_state.csv", delimiter=",", usecols=[0, 1, 4, 5, 7, 8, 9, 10, 47]
    )
    vel = kinematic[:, 8]
    min_time_stamp = max(
        [
            kinematic[0, 0] + 1e-9 * kinematic[0, 1],
            accel_with_timestamp[0, 0] + 1e-9 * accel_with_timestamp[0, 1],
            brake_with_timestamp[0, 0] + 1e-9 * brake_with_timestamp[0, 1],
        ]
    )
    max_time_stamp = min(
        [
            kinematic[-1, 0] + 1e-9 * kinematic[-1, 1],
            accel_with_timestamp[-1, 0] + 1e-9 * accel_with_timestamp[-1, 1],
            brake_with_timestamp[-1, 0] + 1e-9 * brake_with_timestamp[-1, 1],
        ]
    )
    data_num = int((max_time_stamp - min_time_stamp)/control_dt)
    data_time_stamps = min_time_stamp + control_dt * np.arange(data_num)
    vel_interp = scipy.interpolate.interp1d(kinematic[:, 0] + 1e-9 * kinematic[:, 1], vel)(data_time_stamps)
    accel_interp = scipy.interpolate.interp1d(accel_with_timestamp[:, 0] + 1e-9 * accel_with_timestamp[:, 1], accel_with_timestamp[:,2])(data_time_stamps) 
    brake_interp = scipy.interpolate.interp1d(brake_with_timestamp[:, 0] + 1e-9 * brake_with_timestamp[:, 1], brake_with_timestamp[:,2])(data_time_stamps)
    acc_interp = np.zeros((data_num,3))
    for i in range(data_num):
        t_sec = int(data_time_stamps[i])
        t_n_sec = int(1e9 * (data_time_stamps[i] - t_sec))
        acc_interp[i,0] = t_sec
        acc_interp[i,1] = t_n_sec
        if accel_interp[i] > 0:
            acc_interp[i,2] = accel_map_control.get_sim_actuation(vel_interp[i], accel_interp[i])
        else:
            acc_interp[i,2] = brake_map_control.get_sim_actuation(vel_interp[i], brake_interp[i])
    return acc_interp
