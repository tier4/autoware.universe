import math


def calculate_kinematics(start_pos, start_speed, reaction_time, jerk, target_accel):
    # After reaction time
    pos_after_reaction = start_pos + start_speed * reaction_time
    speed_after_reaction = start_speed
    time_after_reaction = reaction_time

    # Time to reach target acceleration
    time_to_reach_target_accel = target_accel / jerk
    pos_after_target_accel = pos_after_reaction + speed_after_reaction * \
        time_to_reach_target_accel + (1.0/6.0) * jerk * time_to_reach_target_accel**3
    speed_after_target_accel = speed_after_reaction + 0.5 * jerk * time_to_reach_target_accel ** 2

    # vehicle does not go in reverse and stops before target acc is reached
    if speed_after_target_accel < 0.0:
        time_to_zero_speed = math.sqrt(- 2 * speed_after_reaction / jerk)
        pos_when_zero_speed = pos_after_reaction + speed_after_reaction * \
            time_to_zero_speed + (1.0/6.0) * jerk * time_to_zero_speed**3

        time_when_zero_speed = reaction_time + time_to_zero_speed
        results = {
            "position_after_reaction_time": pos_after_reaction,
            "time_after_reaction_time": time_after_reaction,
            "position_when_target_acceleration_reached": None,
            "time_when_target_acceleration_reached": None,
            "position_when_zero_speed_reached": pos_when_zero_speed,
            "time_when_zero_speed_reached": time_when_zero_speed
        }
        return results

    time_after_target_accel = reaction_time + time_to_reach_target_accel
    # Time to reach 0 speed after target acceleration is reached
    time_to_zero_speed = speed_after_target_accel / -target_accel
    pos_when_zero_speed = pos_after_target_accel + speed_after_target_accel * \
        time_to_zero_speed + 0.5 * target_accel * time_to_zero_speed**2
    time_when_zero_speed = time_after_target_accel + time_to_zero_speed

    results = {
        "position_after_reaction_time": pos_after_reaction,
        "time_after_reaction_time": time_after_reaction,
        "position_when_target_acceleration_reached": pos_after_target_accel,
        "time_when_target_acceleration_reached": time_after_target_accel,
        "position_when_zero_speed_reached": pos_when_zero_speed,
        "time_when_zero_speed_reached": time_when_zero_speed
    }

    return results


# Example usage
start_pos = 0.0        # Starting position in meters
start_speed = 20.0     # Starting speed in m/s
reaction_time = 1.0    # Reaction time in seconds
jerk = -2.0             # Jerk in m/s^3
target_accel = -4.0    # Target acceleration in m/s^2

results = calculate_kinematics(start_pos, start_speed, reaction_time, jerk, target_accel)
for key, value in results.items():
    print(f"{key}: {value:.2f}")
