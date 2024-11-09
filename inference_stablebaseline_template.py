import os
import sys
import time
import cv2
import numpy as np
from stable_baselines3 import PPO

from CarRLEnvironment import CarRLEnvironment
from train_stable_baseline import CustomCNN
from CarDataService import CarSocketService, CarData


def _preprocess_observation( image):
    """
    Preprocess the image for observation by resizing and converting it to grayscale.

    Args:
        image (numpy.ndarray): The original image from the car's camera.

    Returns:
        processed_image (numpy.ndarray): The processed grayscale image.
    """
    resized_image = cv2.resize(image, (64, 64))
    grayscale_image = np.mean(resized_image, axis=2, keepdims=True)
    return grayscale_image.astype(np.uint8)


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
    current_observation = {
        "image": _preprocess_observation(car_data.image),
        "steering_speed": np.array([car_data.steering_angle, car_data.speed], dtype=np.float32)
    }

    # Use the model's predict method to get the steering angle and throttle
    steering_angle, throttle = rl_model.predict(current_observation, deterministic=True)[0]

    # Check if a reset is triggered based on the car data
    reset_trigger = should_reset(car_data)

    return steering_angle, throttle, reset_trigger


if __name__ == '__main__':
    # Initialize the car service with a system delay (adjust for performance)
    # system_delay: It is recommended to set 0.1 seconds
    car_service = CarSocketService(system_delay=0.1)

    policy_kwargs = {
        "features_extractor_class": CustomCNN,
        "features_extractor_kwargs": {"features_dim": 256},  # Change feature dimensions if needed
    }
    # env = CarRLEnvironment(car_service)  # Adjust frame_stack_num based on how many frames to stack
    rl_model = PPO.load(r".\PPO_best_model.zip")

    # Start the car service with the RL process
    car_service.start_with_RLProcess(RL_Process)
