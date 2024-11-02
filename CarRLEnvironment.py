import time
from collections import deque
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import eventlet
from CarDataService import CarSocketService, CarData


class CarRLEnvironment(gym.Env):
    def __init__(self, car_service: CarSocketService):
        """
        Initialize the CarRL environment with a given car service and number of frames to stack.

        Args:
            car_service (CarSocketService): The service that communicates with the car's simulator.
            frame_stack_num (int): Number of frames to stack for observation space.
        """
        super(CarRLEnvironment, self).__init__()

        self.car_service = car_service
        self.car_service.start_with_nothing()

        # Observation space includes stacked frames and steering/speed information.
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(64, 64, 1), dtype=np.uint8),
            "steering_speed": spaces.Box(low=np.array([-25.0, 0.0]), high=np.array([25.0, 100.0]), dtype=np.float32)
        })

        # Action space: steering angle (-1 to 1) and throttle (0 to 1)
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)

        # Initialize observation and other variables
        self.current_observation = {
            "image": np.zeros(self.observation_space['image'].shape, dtype=np.uint8),
            "steering_speed": np.zeros(self.observation_space['steering_speed'].shape, dtype=np.float32)
        }
        self.done = False
        self._last_timestamp = 0
        self.start_time = None
        self.system_delay = car_service.system_delay
        self.progress_queue = deque(maxlen=5)
        self.__check_done_use_last_timestamp = 0
        self.__check_done_use_progress = 0

        # Wait for connection and data
        while not (self.car_service.client_connected and self.car_service.initial_data_received):
            eventlet.sleep(self.system_delay)

    def reset(self, seed=None, options=None):
        """
        Reset the environment for a new episode.

        Args:
            seed: Optional random seed for environment reset.
            options: Optional additional parameters.

        Returns:
            observation (dict): The initial observation containing the stacked images and steering/speed.
            info (dict): Additional information (empty in this case).
        """
        self.done = False
        self.car_service.send_control(0, 0, True)  # Send stop command for a clean reset
        self.car_service.wait_for_new_data()

        car_data = self.car_service.carData

        # Preprocess the image and initialize the frame history
        image = car_data.image if car_data.image is not None else np.zeros((64, 64, 3), dtype=np.float32)
        processed_image = self._preprocess_observation(image)

        # Initialize observation with steering and speed
        self.current_observation = {
            "image": processed_image,
            "steering_speed": np.array([0.0, 0.0], dtype=np.float32)
        }

        self.start_time = time.time()
        self._last_timestamp = car_data.timestamp
        self.progress_queue.clear()
        self.__check_done_use_last_timestamp = car_data.timestamp
        self.__check_done_use_progress = 0

        return self.current_observation, {}

    def step(self, action):
        """
        Execute one step in the environment with the given action.

        Args:
            action (array): The action containing [steering_angle, throttle].

        Returns:
            observation (dict): Updated observation with stacked images and steering/speed data.
            reward (float): The reward for the step.
            done (bool): Whether the episode has ended.
            truncated (bool): Whether the episode is truncated (always False here).
            info (dict): Additional info (empty in this case).
        """
        # DO NOT CHANGE THE FOLLOWING CODE
        steering_angle, throttle = action
        self.car_service.send_control(steering_angle, throttle)
        self.car_service.wait_for_new_data()
        # DO NOT CHANGE THE PREVIOUS CODE

        car_data = self.car_service.carData
        self.progress_queue.append(float(car_data.progress))

        # Process and stack images
        image = car_data.image if car_data.image is not None else np.zeros((64, 64, 3), dtype=np.float32)
        processed_image = self._preprocess_observation(image)

        current_steering = float(car_data.steering_angle)
        current_speed = min(float(car_data.speed), 100.0)

        self.current_observation = {
            "image": processed_image,
            "steering_speed": np.array([current_steering, current_speed], dtype=np.float32)
        }

        reward = self._compute_reward(car_data)
        self.done = self._check_done(car_data)

        # Update timestamp and calculate FPS
        time_diff = self.car_service.carData.timestamp - self._last_timestamp
        fps = int(1000 / time_diff) if time_diff > 0 else 0
        print(f"\r{fps: 03} fps, reward: {reward: 05.2f}", end="")
        self._last_timestamp = car_data.timestamp

        return self.current_observation, reward, self.done, False, {}

    def _compute_reward(self, car_data: CarData):
        """
        Compute the reward for the current step based on the car's progress and position.

        Args:
            car_data (CarData): The current car data received from the car service.

        Returns:
            reward (float): The calculated reward based on progress and track position.
        """
        reward = (self.progress_queue[-1] - self.progress_queue[0]) * 100 + car_data.velocity_z * 0.005
        if car_data.y < 0:
            reward -= 10  # Penalize if off track
        if car_data.obstacle_car == 1:
            reward -= 10  # Penalize if there is an obstacle
        return reward

    def _check_done(self, car_data: CarData):
        """
        Check if the episode is done based on the car's position or progress.

        Args:
            car_data (CarData): The current car data received from the car service.

        Returns:
            done (bool): Whether the episode is finished.
        """
        if car_data.y < 0 or car_data.progress >= 100.0:
            return True

        if car_data.timestamp - self.__check_done_use_last_timestamp > 30_000 / car_data.time_speed_up_scale:
            if car_data.progress - self.__check_done_use_progress < 0.001:
                return True
            self.__check_done_use_last_timestamp = car_data.timestamp
            self.__check_done_use_progress = car_data.progress

        return False

    def _preprocess_observation(self, image):
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

    def render(self, mode="human"):
        """
        Render the current car camera view (used for debugging).

        Args:
            mode (str): The render mode (default is 'human').
        """
        if self.car_service.carData.image is not None:
            cv2.imshow("Car Camera", self.car_service.carData.image)
            cv2.waitKey(1)

    def close(self):
        """
        Clean up any resources (e.g., close OpenCV windows).
        """
        cv2.destroyAllWindows()
