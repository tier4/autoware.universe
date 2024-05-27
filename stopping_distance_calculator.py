import math
import pandas as pd
import numpy as np


class Params:
    def __init__(self, pos=0, speed=0, reaction_time=0, accel=0):
        self.pos = pos
        self.speed = speed
        self.reaction_time = reaction_time
        self.accel = accel


def calculate_collision_with_rss(ego_params: Params, front_vehicle_params: Params, rear_vehicle_params: Params):
    ego_pos, ego_speed, rt, ego_acc = ego_params.pos, ego_params.speed, ego_params.reaction_time, ego_params.accel

    front_pos, front_speed, _, front_acc = front_vehicle_params.pos, front_vehicle_params.speed, front_vehicle_params.reaction_time, front_vehicle_params.accel
    is_front_collision = front_pos - ego_pos < ego_speed * rt + \
        (ego_speed ** 2 / (-2.0 * ego_acc)) - (front_speed ** 2) / (-2*front_acc)

    rear_pos, rear_speed, rt, rear_acc = rear_vehicle_params.pos, rear_vehicle_params.speed, rear_vehicle_params.reaction_time, rear_vehicle_params.accel
    is_rear_collision = ego_pos - rear_pos < rear_speed * rt + \
        (rear_speed ** 2 / (-2.0 * rear_acc)) - (ego_speed ** 2) / (-2*ego_acc)

    return is_front_collision, is_rear_collision


# Set the parameters for front and rear vehicles
front_vehicle_params = Params(pos=30, speed=10, reaction_time=1.5, accel=-2.0)
rear_vehicle_params = Params(pos=-30, speed=10, reaction_time=1.5, accel=-2.0)

# Generate the data for the table
speeds = np.arange(1, 31)  # 1 m/s to 30 m/s
accelerations = np.linspace(-0.5, -10, 20)  # -0.5 m/s² to -10 m/s²

collision_table = pd.DataFrame(index=accelerations, columns=speeds)

for speed in speeds:
    for accel in accelerations:
        ego_params = Params(pos=0, speed=speed, reaction_time=1.5, accel=accel)
        is_front_collision, is_rear_collision = calculate_collision_with_rss(
            ego_params, front_vehicle_params, rear_vehicle_params)
        if is_front_collision and is_rear_collision:
            collision_table.at[accel, speed] = "Both"
        elif is_front_collision:
            collision_table.at[accel, speed] = "Front"
        elif is_rear_collision:
            collision_table.at[accel, speed] = "Rear"
        else:
            collision_table.at[accel, speed] = "None"

print(collision_table)
