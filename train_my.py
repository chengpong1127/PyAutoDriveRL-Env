import os
import time
import cv2
import numpy as np
from CarDataService import CarSocketService, CarData


class CarRLController:
    """
    This class handles the interaction between car simulation data and reinforcement learning logic.
    It processes car data, computes rewards, and selects actions for the car based on RL algorithms.
    """

    def __init__(self, car_service: CarSocketService):
        """
        Initialize the CarRLController.
        """
        self.car_service = car_service
        self.last_time = time.time()  # To track and print time intervals for debugging
        self.last_progress = 0.0
        # self.model

    def process_car_data(self, car_data):
        """
        Process car data, compute rewards, and return actions.

        Args:
            car_data (CarData): Latest car telemetry and camera data.

        Returns:
            tuple: (steering_angle, throttle, reset_trigger)
                * steering_angle (float): The steering angle (-1 to 1).
                * throttle (float): The throttle value (-1 to 1), less than 0 to reverse and brake.
                * reset_trigger (bool): True if the car should be reset, False otherwise.
        """
        # ===== debug message =====
        self._clear_console()
        self._print_debug_info(car_data)  # Debugging info for telemetry and time intervals
        # =========================

        # Step 1: Extract observation data from car_data
        obs = self._extract_observations(car_data)

        # Step 2: Compute the reward based on the car data, and train the model
        reward = self._compute_reward(car_data)
        # model.train(obs, reward)

        # Step 3: Select actions based on RL model (steering and throttle)
        steering_angle, throttle = self._select_action(obs)

        # Step 4: Check if the car should be reset
        reset_trigger = self._should_reset(car_data)

        # Step 5: Show car's camera view (for visualization, can be removed if not needed)
        # self._visualize(car_data.image)

        # ===== update =====
        self.last_progress = car_data.progress
        self.last_time = car_data.timestamp

        return steering_angle, throttle, reset_trigger

    def _extract_observations(self, car_data) -> tuple:
        """
        Extract key observation data from car_data.

        Args:
            car_data (CarData): Contains telemetry information like progress, speed, etc.

        Returns:
            tuple: (progress, speed, acceleration_z, velocity_z)
        """
        return car_data.progress, car_data.speed, car_data.acceleration_z, car_data.velocity_z

    def _compute_reward(self, car_data):
        """
        Compute reward based on the car's performance.

        Args:
            car_data (CarData): Latest car telemetry and camera data.

        Returns:
            float: The computed reward value.
        """
        return (car_data.progress - self.last_progress) * 100 + car_data.velocity_z * 0.01

    def _select_action(self, obs: tuple):
        """
        Use the RL model to select the car's steering and throttle actions.
        Currently uses random values but can be replaced with an actual RL model.

        Args:
            obs (tuple): List of observation values.

        Returns:
            tuple: (steering_angle, throttle)
        """
        # Example logic: Use random values for demonstration purposes (replace with RL model logic)
        steering_angle = np.random.uniform(-1, 1)
        throttle = np.random.uniform(-1, 1)

        # Replace with actual RL model inference
        # steering_angle, throttle = model.predict(obs)

        return steering_angle, throttle

    def _should_reset(self, car_data):
        """
        Determine if the car needs to be reset.

        Args:
            car_data (CarData): Contains telemetry information such as progress and car's position.

        Returns:
            int: 1 if the car needs to be reset, 0 otherwise.
        """

        if car_data.y < 0 or car_data.progress >= 1:
            return 1
        else:
            return 0

    def _visualize(self, image):
        """
        Display the car's front camera view using OpenCV.

        Args:
            image (numpy.ndarray): The car's camera image.
        """
        cv2.imshow("Car Camera View", image)
        cv2.waitKey(1)

    def _clear_console(self):
        """
        Clear the console output for easier debugging.
        """
        os.system('cls' if os.name == 'nt' else 'clear')

    def _print_debug_info(self, car_data):
        """
        Print car telemetry and frame time interval for debugging purposes.

        Args:
            car_data (CarData): Contains telemetry information like speed, progress, etc.
        """
        print(car_data)
        print(f"Time since last frame: {car_data.timestamp - self.last_time} ms, fps: {int(1000 / (car_data.timestamp - self.last_time))}")


if __name__ == '__main__':
    """
    Main entry point for the car RL control loop.
    This section connects the car's socket service with the RL logic.
    """

    # Initialize the CarSocketService with a system delay (adjust for performance)
    # system_delay: It is recommended to set 0.1 seconds
    car_service = CarSocketService(system_delay=0.1)

    # Create an instance of the RL controller
    car_rl_controller = CarRLController(car_service)

    # Define a wrapper to process car data using the RL controller
    def RL_Process(car_data):
        """
        Wrapper function to process car data and get RL-based actions.

        Args:
            car_data (CarData): Contains the latest telemetry and image data.

        Returns:
            tuple: (steering_angle, throttle, reset_trigger)
        """
        return car_rl_controller.process_car_data(car_data)


    # Start the car service and continuously process data using RL logic
    car_service.start_with_RLProcess(RL_Process)
