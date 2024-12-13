import time
from collections import deque
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import eventlet
from CarDataService import CarSocketService, CarData
from lane_detection import lane_detection
from transformers import pipeline
from PIL import Image
import os


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
        self.image_size = (120, 60)

        # Observation space includes stacked frames and steering/speed information.
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(self.image_size[1], self.image_size[0], 3), dtype=np.uint8),
            "line_image": spaces.Box(low=0, high=255, shape=(self.image_size[1], self.image_size[0]), dtype=np.uint8),
            "depth_image": spaces.Box(low=0, high=255, shape=(self.image_size[1], self.image_size[0]), dtype=np.uint8),
            "road_segmentation_image": spaces.Box(low=0, high=255, shape=(self.image_size[1], self.image_size[0]), dtype=np.uint8),
            "optical_flow": spaces.Box(low=-np.inf, high=np.inf, shape=(self.image_size[1], self.image_size[0], 2), dtype=np.float32),
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
            "progress_diff": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            "timestamp": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.int64),
            "y": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            "time_speed_up_scale": spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
            "manual_control": spaces.MultiBinary(1),
            "obstacle_car": spaces.MultiBinary(1),
            "steering_angle_queue": spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32),
            "throttle_queue": spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32),
        })

        # Action space: steering angle (-1 to 1) and throttle (0 to 1)
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
        #self.action_space = spaces.MultiDiscrete([9, 4])
        # Initialize observation and other variables
        self.current_observation = None
        self.done = False
        self._last_timestamp = 0
        self.start_time = None
        self.system_delay = car_service.system_delay
        self.progress_queue = deque(maxlen=2)
        self.speed_queue = deque(maxlen=10)
        self.image_queue = deque(maxlen=3)
        self.steering_angle_queue = deque(maxlen=5)
        self.throttle_queue = deque(maxlen=5)
        self.__check_done_use_last_timestamp = 0
        self.__check_done_use_progress = 0
        
        self.depth_model = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device='cuda')
        self.segmentation_model = pipeline("image-segmentation", model="nvidia/segformer-b5-finetuned-ade-640-640", device='cuda')

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
        # Initialize observation with steering and speed
        self.current_observation = self.car_data_to_observation(car_data)

        self.start_time = time.time()
        self._last_timestamp = car_data.timestamp
        self.progress_queue.clear()
        self.speed_queue.clear()
        self.image_queue.clear()
        self.steering_angle_queue.clear()
        self.throttle_queue.clear()
        self.__check_done_use_last_timestamp = car_data.timestamp
        self.__check_done_use_progress = 0
        
        #self.current_throttle = 0.3
        self.current_steering_angle = 0.0

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
        self.speed_queue.append(float(car_data.speed))
        self.image_queue.append(car_data.image)
        self.steering_angle_queue.append(steering_angle)
        self.throttle_queue.append(throttle)

        self.current_observation = self.car_data_to_observation(car_data)

        reward = self._compute_reward(car_data)
        self.done = self._check_done(car_data)

        # Update timestamp and calculate FPS
        time_diff = self.car_service.carData.timestamp - self._last_timestamp
        fps = int(1000 / time_diff) if time_diff > 0 else 0
        self.display_info(fps, fps/car_data.time_speed_up_scale, reward, car_data, action)
        self._last_timestamp = car_data.timestamp

        return self.current_observation, reward, self.done, False, {}
    
    def display_info(self, fps, unity_fps, reward, car_data, action):
        # clear console
        os.system('cls' if os.name == 'nt' else 'clear')
        print(car_data)
        print(f"FPS: {fps: 05.1f} -> Unity World FPS: {unity_fps: 05.1f}, Reward: {reward: 05.2f}")
    

    def _compute_reward(self, car_data: CarData):
        """
        Compute the reward for the current step based on the car's progress and position.

        Args:
            car_data (CarData): The current car data received from the car service.

        Returns:
            reward (float): The calculated reward based on progress and track position.
        """
        reward = (self.progress_queue[-1] - self.progress_queue[0]) * 1000 + car_data.velocity_z * 0.005
        # if len(self.steering_angle_queue) > 1:
        #     reward -= (self.steering_angle_queue[-1] - self.steering_angle_queue[-2]) ** 2
        reward += self.current_observation['road_segmentation_image'].mean() / 256.0 * 0.1
        if car_data.y < 0 or len(self.speed_queue) > 8 and np.mean(self.speed_queue) < 0.1:
            reward = -10
        # if car_data.obstacle_car == 1:
        #     reward -= 0.1  # Penalize if there is an obstacle
        
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

        if car_data.timestamp - self.__check_done_use_last_timestamp > 20_000 / car_data.time_speed_up_scale:
            if car_data.progress - self.__check_done_use_progress < 0.001:
                return True
            self.__check_done_use_last_timestamp = car_data.timestamp
            self.__check_done_use_progress = car_data.progress
        
        if len(self.speed_queue) > 8 and np.mean(self.speed_queue) < 0.1:
            print("Car is stuck!")
            return True

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
    
    def get_optical_flow(self):
        """
        Get optical flow between two images.
        """
        if len(self.image_queue) < 2:
            return np.zeros((480, 960, 2), dtype=np.float32)
        
        pre_image = self.image_queue[0]
        next_image = self.image_queue[-1]
        pre_image = cv2.cvtColor(pre_image, cv2.COLOR_RGB2GRAY)
        next_image = cv2.cvtColor(next_image, cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(pre_image, next_image, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        return flow.astype(np.float32)
        
    def car_data_to_observation(self, car_data):
        """
        Convert CarData instance to observation space dictionary.
        """
        
        depth_image = np.array(self.depth_model(Image.fromarray(car_data.image))['depth'])
        segmentation_result = self.segmentation_model(Image.fromarray(car_data.image))
        road_segmentation = [np.array(result['mask']) for result in segmentation_result if result['label'] in ['road', 'earth', 'building']]
        
        # debug: save segmentation result
        original_image = Image.fromarray(car_data.image)
        # add segmentation result to the image
        for result in segmentation_result:
            mask = np.array(result['mask'])
            mask = np.stack([mask, mask, mask], axis=-1)
            original_image = Image.blend(original_image, Image.fromarray(mask), alpha=0.5)
            
        original_image.save("segmentation_result.jpg")
        
        if len(road_segmentation) > 0:
            # or all the road labels
            road_segmentation = np.sum(np.array(road_segmentation), axis=0, dtype=np.uint8)
        else:
            road_segmentation = np.zeros(car_data.image.shape[:2], dtype=np.uint8)
        
        lane_detection_result = lane_detection(car_data.image)
        
        def resize(image):
            image = image[240: 360, :]
            return cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)
        
        progress_diff = self.progress_queue[-1] - self.progress_queue[0] if len(self.progress_queue) > 1 else 0.0
        steering_angle_queue = list(self.steering_angle_queue)
        while len(steering_angle_queue) < 5:
            steering_angle_queue.append(0.0)
        
        throttle_queue = list(self.throttle_queue)
        while len(throttle_queue) < 5:
            throttle_queue.append(0.0)

        observation = {
            "image": resize(car_data.image),
            "line_image": resize(lane_detection_result["line_image"]),
            "depth_image": resize(depth_image),
            "optical_flow": resize(self.get_optical_flow()),
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
            "steering_angle_queue": np.array(steering_angle_queue, dtype=np.float32),
            "throttle_queue": np.array(throttle_queue, dtype=np.float32),
        }
        
        cv2.imwrite("images/initial_image.jpg", car_data.image)
        cv2.imwrite("images/image.jpg", observation["image"])
        cv2.imwrite("images/line_image.jpg", observation["line_image"])
        cv2.imwrite("images/optical_flow_x.jpg", observation["optical_flow"][:, :, 0])
        cv2.imwrite("images/optical_flow_y.jpg", observation["optical_flow"][:, :, 1])
        cv2.imwrite("images/road_segmentation_image.jpg", observation["road_segmentation_image"])
        
        
        
        
        
        for key, value in observation.items():
            if np.isnan(value).any():
                observation[key] = np.nan_to_num(value)

        return observation
