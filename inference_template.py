import os
import sys
import time
import cv2
import numpy as np
from CarDataService import CarSocketService, CarData


def extract_observations(car_data: CarData):
    return car_data.image


def model(obs):
    return np.random.uniform(-1, 1), np.random.uniform(-1, 1)


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
    obs = extract_observations(car_data)

    # Get the steering angle and throttle predictions from the model
    steering_angle, throttle = model(obs)

    # Check if a reset is triggered based on the car data
    reset_trigger = should_reset(car_data)

    return steering_angle, throttle, reset_trigger


if __name__ == '__main__':
    # Initialize the car service with a system delay (adjust for performance)
    # system_delay: It is recommended to set 0.1 seconds
    car_service = CarSocketService(system_delay=0.1)
    # Start the car service with the RL process
    car_service.start_with_RLProcess(RL_Process)
