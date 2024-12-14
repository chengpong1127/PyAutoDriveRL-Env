import os
import sys
import time
import cv2
import numpy as np
from stable_baselines3 import PPO
from PIL import Image

from CarRLEnvironment import CarRLEnvironment
from train_stable_baseline import CustomCNN
from CarDataService import CarSocketService, CarData

from lane_detection import lane_detection
from transformers import pipeline
from collections import deque


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
    
    if len(speed_queue) > 8 and np.mean(speed_queue) < 0.1:
        print("Car is stuck!")
        return True
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
    progress_queue.append(car_data.progress)
    current_observation = car_data_to_observation(car_data)

    # Use the model's predict method to get the steering angle and throttle
    steering_angle, throttle = rl_model.predict(current_observation, deterministic=False)[0]

    # Check if a reset is triggered based on the car data
    reset_trigger = should_reset(car_data)

    return steering_angle, throttle, reset_trigger

def car_data_to_observation(car_data):
    """
    Convert CarData instance to observation space dictionary.
    """
    
    depth_image = np.array(depth_model(Image.fromarray(car_data.image))['depth'])
    segmentation_result = segmentation_model(Image.fromarray(car_data.image))
    road_segmentation = [np.array(result['mask']) for result in segmentation_result if result['label'] in ['road', 'earth', 'building']]
    
    gray_image = cv2.cvtColor(car_data.image, cv2.COLOR_RGB2GRAY)
    gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    if len(road_segmentation) > 0:
        # or all the road labels
        road_segmentation = np.sum(np.array(road_segmentation), axis=0, dtype=np.uint8)
    else:
        road_segmentation = np.zeros(car_data.image.shape[:2], dtype=np.uint8)
    
    lane_detection_result = lane_detection(car_data.image)
    
    def resize(image):
        image = image[180: 360, :]
        return cv2.resize(image, (64, 64), interpolation=cv2.INTER_LINEAR)
    
    progress_diff = progress_queue[-1] - progress_queue[0] if len(progress_queue) > 1 else 0.0

    observation = {
        "image": resize(car_data.image),
        "gray_image": resize(gray_image),
        "line_image": resize(lane_detection_result["line_image"]),
        "depth_image": resize(depth_image),
        "optical_flow": np.zeros((64, 64, 2), dtype=np.float32),
        "road_segmentation_image": resize(road_segmentation),
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
        "progress_diff": np.array([progress_diff], dtype=np.float32),
        "timestamp": np.array([car_data.timestamp], dtype=np.int64),
        "y": np.array([car_data.y], dtype=np.float32),
        "time_speed_up_scale": np.array([car_data.time_speed_up_scale], dtype=np.float32),
        "manual_control": np.array([car_data.manual_control], dtype=np.int8),
        "obstacle_car": np.array([car_data.obstacle_car], dtype=np.int8),
        "steering_angle_queue": np.zeros(5, dtype=np.float32),
        "throttle_queue": np.zeros(5, dtype=np.float32),
    }
    
    # check road_segmentation_image 255 count, if left > right, direction = -1, else 1
    left = observation["road_segmentation_image"][:, :30].sum()
    right = observation["road_segmentation_image"][:, -30:].sum()
    if left > right:
        observation["direction"] = np.array([-1.0], dtype=np.float32)
    elif right > left:
        observation["direction"] = np.array([1.0], dtype=np.float32)
    else:
        observation["direction"] = np.array([0.0], dtype=np.float32)
    
    
    for key, value in observation.items():
        if np.isnan(value).any():
            observation[key] = np.nan_to_num(value)

    return observation


if __name__ == '__main__':
    # Initialize the car service with a system delay (adjust for performance)
    # system_delay: It is recommended to set 0.1 seconds
    depth_model = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device='cuda')
    segmentation_model = pipeline("image-segmentation", model="nvidia/segformer-b5-finetuned-ade-640-640", device='cuda')
    progress_queue = deque(maxlen=2)
    speed_queue = deque(maxlen=10)
    car_service = CarSocketService(system_delay=0.1)

    policy_kwargs = {
        "features_extractor_class": CustomCNN,
        "features_extractor_kwargs": {"features_dim": 32},  # Change feature dimensions if needed
        "net_arch": dict(pi=[32], vf=[16])
    }
    # env = CarRLEnvironment(car_service)  # Adjust frame_stack_num based on how many frames to stack
    rl_model = PPO.load(r"./result2/model.zip")

    # Start the car service with the RL process
    car_service.start_with_RLProcess(RL_Process)
