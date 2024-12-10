import os
import sys
import time
import cv2
import numpy as np
from CarDataService import CarSocketService, CarData

import os

import numpy as np
from stable_baselines3 import PPO

from CarRLEnvironment import CarRLEnvironment
from train_stable_baseline import CustomCNN
from CarDataService import CarSocketService

def extract_observations(car_data: CarData):
    return car_data.image

# Choose between SAC or PPO model (PPO used here for example)
# model = SAC("MultiInputPolicy", env, policy_kwargs=policy_kwargs, buffer_size=2048, verbose=1, batch_size=64)

def car_data_to_observation(car_data):
    """
    Convert CarData instance to observation space dictionary.
    """

    observation = {
        "image": car_data.image,
        "steering_angle": np.array([car_data.steering_angle], dtype=np.float32),
        "throttle": np.array([car_data.throttle], dtype=np.float32),
        "speed": np.array([car_data.speed], dtype=np.float32),
        "velocity": np.array([car_data.velocity_x, car_data.velocity_y, car_data.velocity_z], dtype=np.float32),
        "acceleration": np.array([car_data.acceleration_x, car_data.acceleration_y, car_data.acceleration_z], dtype=np.float32),
        "angular_velocity": np.array([car_data.angular_velocity_x, car_data.angular_velocity_y, car_data.angular_velocity_z], dtype=np.float32),
        "wheel_friction": np.array([car_data.wheel_friction_forward, car_data.wheel_friction_sideways], dtype=np.float32),
        "orientation": np.array([car_data.yaw, car_data.pitch, car_data.roll], dtype=np.float32),
        "brake_input": np.array([car_data.brake_input], dtype=np.float32),
        "progress": np.array([car_data.progress], dtype=np.float32),
        "timestamp": np.array([car_data.timestamp], dtype=np.int64),
        "y": np.array([car_data.y], dtype=np.float32),
        "time_speed_up_scale": np.array([car_data.time_speed_up_scale], dtype=np.float32),
        "manual_control": np.array([car_data.manual_control], dtype=np.int8),
        "obstacle_car": np.array([car_data.obstacle_car], dtype=np.int8)
    }
    
    for key, value in observation.items():
        if np.isnan(value).any():
            observation[key] = np.nan_to_num(value)

    return observation


def should_reset(car_data: CarData):
    if car_data.y < 0 or car_data.progress >= 1:
        return 1
    else:
        return 0


def RL_Process(car_data: CarData):
    """
    Process the car data to determine the steering angle, throttle, and reset trigger.

    Args:
        car_data (CarData): The data received from the car.

    Returns:
        tuple: A tuple containing the steering angle, throttle, and reset trigger.
    """
    sys.stdout.write("\033[H\033[J")
    print(car_data)

    # Extract observations from the car data

    # Get the steering angle and throttle predictions from the model
    obs = car_data_to_observation(car_data)
    steering_angle, throttle = rl_model.predict(obs, deterministic=True)[0]

    # Check if a reset is triggered based on the car data
    reset_trigger = should_reset(car_data)

    return steering_angle, throttle, reset_trigger


if __name__ == '__main__':
    # Initialize the car service with a system delay (adjust for performance)
    # system_delay: It is recommended to set 0.1 seconds
    car_service = CarSocketService(system_delay=0.1)
    env = CarRLEnvironment(car_service)
    policy_kwargs = {
        "features_extractor_class": CustomCNN,
        "features_extractor_kwargs": {"features_dim": 256},  # Change feature dimensions if needed
    }
    rl_model = PPO.load(r"PPO_best_model.zip")
    # Start the car service with the RL process
    car_service.start_with_RLProcess(RL_Process)
