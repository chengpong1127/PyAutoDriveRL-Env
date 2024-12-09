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
            "image": spaces.Box(low=0, high=255, shape=(480, 960, 3), dtype=np.uint8),
            "steering_angle": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            "throttle": spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
            "speed": spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
            "velocity": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "acceleration": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "angular_velocity": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "wheel_friction": spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
            "orientation": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "brake_input": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "progress": spaces.Box(low=0.0, high=100.0, shape=(1,), dtype=np.float32),
            "timestamp": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.int32),
            "y": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            "time_speed_up_scale": spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
            "manual_control": spaces.MultiBinary(1),
            "obstacle_car": spaces.MultiBinary(1)
        })

        # Action space: steering angle (-1 to 1) and throttle (0 to 1)
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)

        # Initialize observation and other variables
        self.current_observation = None
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
        self.car_service.send_control(0, 0, 1)  # Send stop command for a clean reset
        self.car_service.wait_for_new_data()

        car_data = self.car_service.carData
        print(car_data)
        # Initialize observation with steering and speed
        self.current_observation = self.car_data_to_observation(car_data)

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

        self.current_observation = self.car_data_to_observation(car_data)

        reward = self._compute_reward(car_data)
        self.done = self._check_done(car_data)

        # Update timestamp and calculate FPS
        time_diff = self.car_service.carData.timestamp - self._last_timestamp
        fps = int(1000 / time_diff) if time_diff > 0 else 0
        print(f"\r{fps: 05.1f} fps -> unity world {fps/car_data.time_speed_up_scale: 05.1f} fps, reward: {reward: 05.2f}", end="")
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
        reward = (self.progress_queue[-1] - self.progress_queue[0]) * 1000 + car_data.velocity_z * 0.005
        if car_data.y < 0:
            reward -= 10  # Penalize if off track
        # if car_data.obstacle_car == 1:
        #     reward -= 0.01  # Penalize if there is an obstacle
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
        
    def car_data_to_observation(self, car_data):
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
            "timestamp": np.array([car_data.timestamp], dtype=np.int32),
            "y": np.array([car_data.y], dtype=np.float32),
            "time_speed_up_scale": np.array([car_data.time_speed_up_scale], dtype=np.float32),
            "manual_control": np.array([car_data.manual_control], dtype=np.int8),
            "obstacle_car": np.array([car_data.obstacle_car], dtype=np.int8)
        }
        
        for key, value in observation.items():
            if np.isnan(value).any():
                observation[key] = np.nan_to_num(value)

        return observation
