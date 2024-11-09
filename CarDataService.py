import base64
import os
import signal
import time

import numpy as np
import cv2
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO
import socketio


class CarData:
    def __init__(self):
        """
        Initialize CarData with all telemetry-related fields.
        """
        self.image = None
        self.steering_angle = np.nan
        self.throttle = np.nan
        self.speed = np.nan
        self.velocity_x = np.nan
        self.velocity_y = np.nan
        self.velocity_z = np.nan
        self.acceleration_x = np.nan
        self.acceleration_y = np.nan
        self.acceleration_z = np.nan
        self.angular_velocity_x = np.nan
        self.angular_velocity_y = np.nan
        self.angular_velocity_z = np.nan
        self.wheel_friction_forward = np.nan
        self.wheel_friction_sideways = np.nan
        self.brake_input = np.nan
        self.progress = np.nan
        self.timestamp = np.nan
        self.yaw = np.nan
        self.pitch = np.nan
        self.roll = np.nan
        self.y = np.nan
        self.time_speed_up_scale = 1
        self.manual_control = 0
        self.obstacle_car = 0

    def update(self, data):
        """
        Updates CarData object with the provided telemetry data.
        """
        self.image = np.asarray(Image.open(BytesIO(base64.b64decode(data["image"]))))[..., ::-1]
        self.steering_angle = float(data["steering_angle"]) if data["steering_angle"] != "N/A" else np.nan
        self.throttle = float(data["throttle"]) if data["throttle"] != "N/A" else np.nan
        self.speed = float(data["speed"]) if data["speed"] != "N/A" else np.nan
        self.velocity_x = float(data.get("velocity_x", "N/A")) if data.get("velocity_x", "N/A") != "N/A" else np.nan
        self.velocity_y = float(data.get("velocity_y", "N/A")) if data.get("velocity_y", "N/A") != "N/A" else np.nan
        self.velocity_z = float(data.get("velocity_z", "N/A")) if data.get("velocity_z", "N/A") != "N/A" else np.nan
        self.acceleration_x = float(data.get("acceleration_x", "N/A")) if data.get("acceleration_x", "N/A") != "N/A" else np.nan
        self.acceleration_y = float(data.get("acceleration_y", "N/A")) if data.get("acceleration_y", "N/A") != "N/A" else np.nan
        self.acceleration_z = float(data.get("acceleration_z", "N/A")) if data.get("acceleration_z", "N/A") != "N/A" else np.nan
        self.angular_velocity_x = float(data.get("angular_velocity_x", "N/A")) if data.get("angular_velocity_x", "N/A") != "N/A" else np.nan
        self.angular_velocity_y = float(data.get("angular_velocity_y", "N/A")) if data.get("angular_velocity_y", "N/A") != "N/A" else np.nan
        self.angular_velocity_z = float(data.get("angular_velocity_z", "N/A")) if data.get("angular_velocity_z", "N/A") != "N/A" else np.nan
        self.wheel_friction_forward = float(data.get("wheel_friction_forward", "N/A")) if data.get("wheel_friction_forward", "N/A") != "N/A" else np.nan
        self.wheel_friction_sideways = float(data.get("wheel_friction_sideways", "N/A")) if data.get("wheel_friction_sideways", "N/A") != "N/A" else np.nan
        self.brake_input = float(data.get("brake_input", "N/A")) if data.get("brake_input", "N/A") != "N/A" else np.nan
        self.progress = float(data.get("progress", "N/A")) if data.get("progress", "N/A") != "N/A" else np.nan
        self.timestamp = int(data.get("timestamp", "N/A")) if data.get("timestamp", "N/A") != "N/A" else np.nan
        self.yaw = float(data.get("yaw", "N/A")) if data.get("yaw", "N/A") != "N/A" else np.nan
        self.pitch = float(data.get("pitch", "N/A")) if data.get("pitch", "N/A") != "N/A" else np.nan
        self.roll = float(data.get("roll", "N/A")) if data.get("roll", "N/A") != "N/A" else np.nan
        self.y = float(data.get("y", "N/A")) if data.get("y", "N/A") != "N/A" else np.nan
        self.time_speed_up_scale = float(data.get("time_speed_up_scale", "N/A")) if data.get("time_speed_up_scale", "N/A") != "N/A" else np.nan
        self.manual_control = int(data.get("manual_control", "0")) if data.get("manual_control", "N/A") != "N/A" else 0
        self.obstacle_car = int(data.get("obstacle_car", "0")) if data.get("manual_control", "N/A") != "N/A" else 0

    def __str__(self):
        """
        Automatically generates a string representation of the CarData object
        by iterating over all its attributes in a formatted manner.
        """
        result = []
        result.append(f"{'CarData':^65}")
        result.append("=" * 65)

        result.append(f"{'Timestamp':<30} {self.timestamp}")
        if self.image is not None:
            result.append(f"{'Image shape:':<30} {self.image.shape}")
        else:
            result.append(f"{'Image shape:':<30} None")

        # Format all floating-point numbers to (-)000.0000 format
        result.append(f"{'Steering Angle:':<30} {self.steering_angle: 09.4f}")
        result.append(f"{'Throttle:':<30} {self.throttle: 09.4f}")
        result.append(f"{'Speed:':<30} {self.speed: 09.4f}")

        result.append(
            f"{'Velocity (X, Y, Z):':<30} {self.velocity_x: 09.4f}, {self.velocity_y: 09.4f}, {self.velocity_z: 09.4f}")
        result.append(
            f"{'Acceleration (X, Y, Z):':<30} {self.acceleration_x: 09.4f}, {self.acceleration_y: 09.4f}, {self.acceleration_z: 09.4f}")

        result.append(
            f"{'Angular Velocity (X, Y, Z):':<30} {self.angular_velocity_x: 09.4f}, {self.angular_velocity_y: 09.4f}, {self.angular_velocity_z: 09.4f}")

        result.append(
            f"{'Wheel Friction (F, S):':<30} {self.wheel_friction_forward: 09.4f}, {self.wheel_friction_sideways: 09.4f}")
        result.append(f"{'Brake Input:':<30} {self.brake_input: 09.4f}")

        result.append(f"{'Yaw, Pitch, Roll:':<30} {self.yaw: 09.4f}, {self.pitch: 09.4f}, {self.roll: 09.4f}")
        result.append(f"{'Y Position:':<30} {self.y: 09.4f}")
        result.append(f"{'Hit An Obstacle:':<30} {self.obstacle_car: 09.4f}")
        result.append(f"{'Progress:':<30} {self.progress: 09.4f}")
        result.append(f"{'Time Speed Up Scale:':<30} {self.time_speed_up_scale: 09.4f}")

        result.append("=" * 65)

        return "\n".join(result)


class CarSocketService:
    def __init__(self, sio=None, system_delay=0.1):
        """
        Initialize service and allow setting sleep interval.
        """
        self.sio = sio if sio else socketio.Server()  # Allow passing in an existing sio instance
        self.app = Flask(__name__)
        self.server = None
        self.should_stop = False
        self.carData = CarData()  # Create an instance of CarData class
        self.client_connected = False  # Flag to check if a client is connected
        self.RL_Process = None  # Reinforcement learning process function
        self.system_delay = system_delay   # 在沒加速的情況下的system delay，建議設定0.1、不要太低，不然畫面沒什麼動就要再次決策，只是在浪費資源
        self.debug = False
        self.initial_data_received = False  # To ensure initial data is received
        self.__last_timestamp = 0

        if self.system_delay < 0.05:
            raise ValueError("system_delay to short. Must be > 0.05s")

        self.register_sio_events()

    def wait_for_new_data(self):
        """
        Wait until new car data is available.
        If you do not call this function, you will not be able to receive new data.
        """
        while self.carData.image is None:
            eventlet.sleep(self.system_delay)
        if self.carData.time_speed_up_scale == 1:
            eventlet.sleep(self.system_delay)
        while self.__last_timestamp == self.carData.timestamp:
            eventlet.sleep(0)
        self.__last_timestamp = self.carData.timestamp

    def clear_console(self):
        """Clears the console output for smoother visualization."""
        os.system('cls' if os.name == 'nt' else 'clear')

    def start_with_nothing(self):
        signal.signal(signal.SIGINT, self.signal_handler)

        # Wrap Flask application with engineio's middleware
        self.app = socketio.Middleware(self.sio, self.app)

        # Start the server in a greenlet (Eventlet coroutine)
        self.server = eventlet.spawn(eventlet.wsgi.server, eventlet.listen(('', 4567)), self.app)
        self.register_sio_events()

        while not self.client_connected and not self.initial_data_received:
            eventlet.sleep(0.1)

    def start_with_RLProcess(self, RL_Process=None):
        """
        Starts the Socket.IO service.
        :param RL_Process: Optional function to provide reinforcement learning-based actions.
        """
        self.RL_Process = RL_Process
        signal.signal(signal.SIGINT, self.signal_handler)

        # Wrap Flask application with engineio's middleware
        self.app = socketio.Middleware(self.sio, self.app)

        # Start the server in a greenlet (Eventlet coroutine)
        self.server = eventlet.spawn(eventlet.wsgi.server, eventlet.listen(('', 4567)), self.app)
        self.register_sio_events()

        while not self.client_connected and not self.initial_data_received:
            eventlet.sleep(self.system_delay)

        self.send_control(0, 0, 1)

        try:
            while not self.should_stop:

                self.wait_for_new_data()

                if self.client_connected and self.RL_Process and self.initial_data_received:
                    action = self.RL_Process(self.carData)  # Call RL Process function
                    steering_angle, throttle, reset_trigger = action
                    self.send_control(steering_angle, throttle, reset_trigger)
                else:
                    self.send_control(0, 0, 0)

        except (KeyboardInterrupt, SystemExit):
            print("Caught exit signal.")
        finally:
            self.shutdown()

    def send_control(self, steering_angle, throttle, reset_trigger=0):
        """
        Sends control signals to the car.

        Args:
            steering_angle (float): The steering angle (-1 to 1).
            throttle (float): The throttle value (-1 to 1), less than 0 to reverse and brake.
            reset_trigger (int):
                If set to 1, the car will be reset to its initial position.
                If set to 1~, reset the car to the specified position
                If set to 0, no position reset will be performed
        """
        if self.debug:
            print(
                f"Sending control - Steering Angle: {steering_angle}, Throttle: {throttle}, Reset Trigger: {reset_trigger}")

        try:
            steering_angle = float(steering_angle)
            throttle = float(throttle)
        except ValueError:
            raise ValueError("Steering angle and throttle values must be floats, but got {} and {}".format(steering_angle, throttle))

        try:
            reset_trigger = int(reset_trigger)
        except ValueError:
            raise ValueError("Reset trigger must be an integer, but got {}".format(reset_trigger))

        if -1 > steering_angle > 1 or -1 > throttle > 1:
            raise ValueError("Steering angle and throttle values must be between -1 and 1, but got {} and {}".format(steering_angle, throttle))

        if reset_trigger < 0:
            raise ValueError("Reset trigger must be >= 0, but got {}".format(reset_trigger))

        self.sio.emit(
            "steer",
            data={
                'steering_angle': str(steering_angle),
                'throttle': str(throttle),
                'reset_trigger': str(reset_trigger),
            },
            skip_sid=True
        )

    def shutdown(self):
        """
        Gracefully shutdown the server.
        """
        self.should_stop = True
        if self.server is not None:
            self.server.kill()
        print("Server shutdown completed.")

    def signal_handler(self, signal, frame):
        """
        Handles Ctrl+C or termination signals.
        """
        self.shutdown()
        exit(0)

    def register_sio_events(self):
        """
        Register all Socket.IO event handlers.
        """

        @self.sio.on('telemetry')
        def telemetry(sid, data):
            """
            Handles telemetry data from the car.
            """
            if data:
                # Update the CarData object with telemetry data
                self.carData.update(data)

                # Display the image (for debugging or visualization purposes)
                if self.debug:
                    cv2.imshow("debug_image", self.carData.image[..., ::-1])
                    cv2.waitKey(1)

                # Set the initial data received flag
                self.initial_data_received = True

                if self.debug:
                    self.clear_console()
                    print("--------------------")
                    print(self.carData)
                    print("--------------------")
            else:
                # DO NOT MODIFY HERE
                self.sio.emit('manual', data={}, skip_sid=True)

        @self.sio.on('connect')
        def connect(sid, environ):
            """
            Handles new client connections.
            """
            print("Client connected:", sid)
            self.client_connected = True  # Set client connected flag
            self.initial_data_received = False  # Reset initial data flag

            self.send_control(0, 0, 0)
            eventlet.sleep(self.system_delay)
            self.send_control(0, 0, 1)
            eventlet.sleep(self.system_delay)

        @self.sio.on('disconnect')
        def disconnect(sid):
            """
            Handles client disconnections.
            """
            print("Client disconnected:", sid)
            self.client_connected = False  # Reset client connected flag
