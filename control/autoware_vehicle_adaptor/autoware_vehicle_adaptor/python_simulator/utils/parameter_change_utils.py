from enum import Enum
import os

import numpy as np
from utils.data_collection_utils import ControlType
from utils.data_collection_utils import DataCollectionMode

CHANGE_PARAM_P1 = 0.8
CHANGE_PARAM_P2 = 1.2

ChangeParam = {}
"""Parameters to be changed when running the simulation."""
ChangeParam["measurement_steer_bias"] = [
        -1.0 * np.pi / 180.0,
        -0.8 * np.pi / 180.0,
        -0.6 * np.pi / 180.0,
        -0.4 * np.pi / 180.0,
        -0.2 * np.pi / 180.0,
        0.0,
        0.2 * np.pi / 180.0,
        0.4 * np.pi / 180.0,
        0.6 * np.pi / 180.0,
        0.8 * np.pi / 180.0,
        1.0 * np.pi / 180.0,
    ]
"""steer midpoint (soft + hard)"""

ChangeParam["steer_rate_lim"] = [0.020, 0.050, 0.100, 0.150, 0.200, 0.300, 0.400, 0.500]
"""Maximum steer angular velocity"""

ChangeParam["vel_rate_lim"] = [0.5, 1.0, 3.0, 5.0, 7.0, 9.0]
"""Maximum acceleration/deceleration"""

ChangeParam["wheel_base"] = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0]
"""wheel base"""

ChangeParam["steer_dead_band"] = [0.0000, 0.0012, 0.0025, 0.0050, 0.01]
"""steer dead band"""

ChangeParam["adaptive_gear_ratio_coef"] = [
        [15.713, 0.053, 0.042, 15.713, 0.053, 0.042],
        [15.713, 0.053, 0.042, CHANGE_PARAM_P1 * 15.713, 0.053, 0.042],
        [15.713, 0.053, 0.042, CHANGE_PARAM_P2 * 15.713, 0.053, 0.042],
        [15.713, 0.053, 0.042, 15.713, CHANGE_PARAM_P1 * 0.053, 0.042],
        [15.713, 0.053, 0.042, CHANGE_PARAM_P1 * 15.713, CHANGE_PARAM_P1 * 0.053, 0.042],
        [15.713, 0.053, 0.042, CHANGE_PARAM_P2 * 15.713, CHANGE_PARAM_P1 * 0.053, 0.042],
        [15.713, 0.053, 0.042, 15.713, CHANGE_PARAM_P2 * 0.053, 0.042],
        [15.713, 0.053, 0.042, CHANGE_PARAM_P1 * 15.713, CHANGE_PARAM_P2 * 0.053, 0.042],
        [15.713, 0.053, 0.042, CHANGE_PARAM_P2 * 15.713, CHANGE_PARAM_P2 * 0.053, 0.042],
        [15.713, 0.053, 0.042, 15.713, 0.053, CHANGE_PARAM_P1 * 0.042],
        [15.713, 0.053, 0.042, CHANGE_PARAM_P1 * 15.713, 0.053, CHANGE_PARAM_P1 * 0.042],
        [15.713, 0.053, 0.042, CHANGE_PARAM_P2 * 15.713, 0.053, CHANGE_PARAM_P1 * 0.042],
        [15.713, 0.053, 0.042, 15.713, CHANGE_PARAM_P1 * 0.053, CHANGE_PARAM_P1 * 0.042],
        [
            15.713,
            0.053,
            0.042,
            CHANGE_PARAM_P1 * 15.713,
            CHANGE_PARAM_P1 * 0.053,
            CHANGE_PARAM_P1 * 0.042,
        ],
        [
            15.713,
            0.053,
            0.042,
            CHANGE_PARAM_P2 * 15.713,
            CHANGE_PARAM_P1 * 0.053,
            CHANGE_PARAM_P1 * 0.042,
        ],
        [15.713, 0.053, 0.042, 15.713, CHANGE_PARAM_P2 * 0.053, CHANGE_PARAM_P1 * 0.042],
        [
            15.713,
            0.053,
            0.042,
            CHANGE_PARAM_P1 * 15.713,
            CHANGE_PARAM_P2 * 0.053,
            CHANGE_PARAM_P1 * 0.042,
        ],
        [
            15.713,
            0.053,
            0.042,
            CHANGE_PARAM_P2 * 15.713,
            CHANGE_PARAM_P2 * 0.053,
            CHANGE_PARAM_P1 * 0.042,
        ],
        [15.713, 0.053, 0.042, 15.713, 0.053, CHANGE_PARAM_P2 * 0.042],
        [15.713, 0.053, 0.042, CHANGE_PARAM_P1 * 15.713, 0.053, CHANGE_PARAM_P2 * 0.042],
        [15.713, 0.053, 0.042, CHANGE_PARAM_P2 * 15.713, 0.053, CHANGE_PARAM_P2 * 0.042],
        [15.713, 0.053, 0.042, 15.713, CHANGE_PARAM_P1 * 0.053, CHANGE_PARAM_P2 * 0.042],
        [
            15.713,
            0.053,
            0.042,
            CHANGE_PARAM_P1 * 15.713,
            CHANGE_PARAM_P1 * 0.053,
            CHANGE_PARAM_P2 * 0.042,
        ],
        [
            15.713,
            0.053,
            0.042,
            CHANGE_PARAM_P2 * 15.713,
            CHANGE_PARAM_P1 * 0.053,
            CHANGE_PARAM_P2 * 0.042,
        ],
        [15.713, 0.053, 0.042, 15.713, CHANGE_PARAM_P2 * 0.053, CHANGE_PARAM_P2 * 0.042],
        [
            15.713,
            0.053,
            0.042,
            CHANGE_PARAM_P1 * 15.713,
            CHANGE_PARAM_P2 * 0.053,
            CHANGE_PARAM_P2 * 0.042,
        ],
        [
            15.713,
            0.053,
            0.042,
            CHANGE_PARAM_P2 * 15.713,
            CHANGE_PARAM_P2 * 0.053,
            CHANGE_PARAM_P2 * 0.042,
        ],
    ]
"""velocity-dependent gear ratio"""

ChangeParam["acc_time_delay"] = [0.00, 0.1, 0.27, 0.40, 0.60, 0.80, 1.01]
"""acc time delay"""

ChangeParam["steer_time_delay"] = [0.00, 0.1, 0.27, 0.40, 0.60, 0.80, 1.02]
"""steer time delay"""

ChangeParam["acc_time_constant"] = [0.01, 0.1, 0.20, 0.24, 0.40, 0.60, 0.80, 1.01]
"""time constant"""

ChangeParam["steer_time_constant"] = [0.01, 0.1, 0.20, 0.24, 0.40, 0.60, 0.80, 1.02]
"""time constant"""

ChangeParam["accel_map_scale"] = [0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
"""pedal - real acceleration correspondence"""

ChangeParam["accel_brake_map_control_path"] = ["../actuation_cmd_maps/accel_brake_maps/low_quality_map"]
"""accel_brake_map_control_path"""

ChangeParam["accel_brake_map_sim_path"] = ["../actuation_cmd_maps/accel_brake_maps/1"]
"""accel_brake_map_sim_path"""

ChangeParam["acc_scaling"] = [0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 5.01]
"""Acceleration scaling coefficient"""

#ChangeParam["steer_scaling"] = [0.9, 0.95, 1.0, 1.05, 1.1]#[0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 5.02]
ChangeParam["steer_scaling"] = [0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 5.02]
"""Steer scaling coefficient"""

ChangeParam["vehicle_type"] = [1, 2, 3, 4]
"""change to other vehicle parameters"""

# ChangeParam["test_vehicle"] = [0, 1, 2, 3, 4, 5, 6, 7]

vehicle_type_params = [
    {},
    {
        "wheel_base": 4.76,
        "acc_time_delay": 1.0,
        "steer_time_delay": 1.0,
        "acc_time_constant": 1.0,
        "steer_time_constant": 1.0,
        "acc_scaling": 0.2,
    },
    {
        "wheel_base": 4.76,
        "acc_time_delay": 0.5,
        "steer_time_delay": 0.5,
        "acc_time_constant": 0.5,
        "steer_time_constant": 0.5,
        "acc_scaling": 0.5,
    },
    {
        "wheel_base": 1.335,
        "acc_time_delay": 0.3,
        "steer_time_delay": 0.3,
        "acc_time_constant": 0.3,
        "steer_time_constant": 0.3,
        "acc_scaling": 1.5,
    },
    {
        "wheel_base": 0.395,
        "acc_time_delay": 0.2,
        "steer_time_delay": 0.2,
        "acc_time_constant": 0.2,
        "steer_time_constant": 0.2,
        "acc_scaling": 3.0,
    },

]

def test_dir_name(
    root: str = ".",
    data_collection_mode: DataCollectionMode | None = None,
    control_type: ControlType | None = None,
    with_adaptor: bool = False,
    change_param: str | None = None,
    index: int | None = None,
    validation_flag: bool = False,
) -> str:
    """Generate string for directory name."""
    dir_name = "log_data/" + root + "/test"
    if control_type is not None:
        dir_name += f"_{control_type.value}_sim"
    elif data_collection_mode is not None:
        dir_name += f"_{data_collection_mode}_aided_sim"
    else:
        dir_name += "_sim"

    if with_adaptor:
        dir_name += "_with_adaptor"
    if change_param is not None:
        dir_name += f"_{change_param}"
    if index is not None:
        dir_name += f"_{index}th"
    if validation_flag:
        dir_name += "_val"
    return dir_name

class DirGenerator:
    """Class to store parameters for `test_dir_name`."""

    def __init__(self, root: str):
        # create directory if not exist
        if not os.path.isdir("log_data"):
            os.mkdir("log_data")
        if not os.path.isdir("log_data/" + root):
            os.mkdir("log_data/" + root)
        self.root = root

    def test_dir_name(
        self,
        data_collection_mode: DataCollectionMode | None = None,
        control_type: ControlType | None = None,
        with_adaptor: bool = False,
        change_param: str | None = None,
        index: int | None = None,
        validation_flag: bool = False,
    ):
        return test_dir_name(
            root=self.root,
            data_collection_mode=data_collection_mode,
            control_type=control_type,
            with_adaptor=with_adaptor,
            change_param=change_param,
            index=index,
            validation_flag=validation_flag,
        )
