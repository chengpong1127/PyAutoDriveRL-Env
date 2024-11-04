# Gym-like Environment Details for CarRLEnvironment

## [CarRLEnvironment.py](https://github.com/Bacon9629/PyAutoDriveRL-Env/blob/main/CarRLEnvironment.py)

The `CarRLEnvironment` class implements a custom Gym-like environment that is specifically tailored for a car simulator,
using `CarSocketService` for communication with the simulation.

### Initialization ([`__init__`](https://github.com/Bacon9629/PyAutoDriveRL-Env/blob/f5275aebf27a09e2f466471be805c2aa247888c0/CarRLEnvironment.py#L12))

- **Car Service Initialization**: The environment relies on an instance of `CarSocketService` to communicate with the
  Unity3D car simulator. It waits for a connection before starting.
- **Observation Space**: This environment uses a dictionary observation space:
    - **Image**: A single grayscale 64x64 image representing the front camera view.
    - **Steering Speed**: A 2D array containing steering angle (between -25 and 25 degrees) and speed (between 0 and
      100).
- **Action Space**: The action space is continuous:
    - **Steering Angle**: A float between -1 (full left) and 1 (full right).
    - **Throttle**: A float between -1 (reverse) and 1 (full throttle forward).

### Reset ([`reset`](https://github.com/Bacon9629/PyAutoDriveRL-Env/blob/f5275aebf27a09e2f466471be805c2aa247888c0/CarRLEnvironment.py#L52))

The `reset` method is responsible for initializing the environment for a new episode. It performs the following steps:

- Sends a stop command to the car simulator.
- Waits for new data to be received from the car simulator.
- Processes the image to be used as an observation.
- Resets various state variables like progress and timestamp to prepare for a new episode.

It returns the initial observation, which is a dictionary of the grayscale image and the initial steering/speed data.

### Step ([`step`](https://github.com/Bacon9629/PyAutoDriveRL-Env/blob/f5275aebf27a09e2f466471be805c2aa247888c0/CarRLEnvironment.py#L88))

The `step` function is called to execute one interaction with the environment based on the selected action. Key
operations:

1. **Action Execution**: The `step` function sends the steering and throttle values to the car simulator.
2. **Observation Update**: It waits for new telemetry data from the car and updates the observations (camera image and
   steering/speed).
3. **Reward Calculation**: Computes a reward based on the car's progress and its position on the track. Off-track
   penalties are applied.
4. **Episode Termination Check**: Determines if the episode should end, based on whether the car went off-track or
   completed the course.
5. **Returns**: The updated observation, reward, whether the episode is done, and additional info.

### Reward Calculation ([`_compute_reward`](https://github.com/Bacon9629/PyAutoDriveRL-Env/blob/f5275aebf27a09e2f466471be805c2aa247888c0/CarRLEnvironment.py#L136))

The reward is calculated primarily based on the car's progress along the track. The environment also penalizes off-track
behavior by reducing the reward if the car's position (`y`) falls below 0.

### Episode Termination ([`_check_done`](https://github.com/Bacon9629/PyAutoDriveRL-Env/blob/f5275aebf27a09e2f466471be805c2aa247888c0/CarRLEnvironment.py#L151))

Episodes are terminated if:

- The car goes off-track (`y < 0`).
- The car completes the track (`progress >= 100`).
- Time-based termination is applied if too much time has passed without significant progress.

### Preprocessing ([`_preprocess_observation`](https://github.com/Bacon9629/PyAutoDriveRL-Env/blob/f5275aebf27a09e2f466471be805c2aa247888c0/CarRLEnvironment.py#L172))

The raw camera images are resized to 64x64 and converted to grayscale for simplicity, which reduces the complexity of
the input data for RL models.

---

## Important Notes (注意事項)

1. **Wait Function Requirement**:
    - The `CarRLEnvironment` uses a waiting mechanism (e.g., `wait_for_new_data()`) to synchronize with the Unity3D car
      simulation. This ensures that the car's telemetry data is up-to-date before taking the next step in the
      environment. Without this wait function, you will be unable to communicate with the Unity3D car simulation

2. **Unity3D Time Acceleration**:
    - If the Unity3D simulation is running in an accelerated mode, be aware that the `timestamp` provided by the
      simulator does **not** reflect this acceleration. It still follows the real-world computer time. This means that
      any time-related functionalities (e.g., time-based episode termination, reward shaping) should be carefully
      designed to account for this discrepancy.
    - If time acceleration is enabled in Unity3D car simulation, you need to adjust your environment’s logic accordingly
      by scaling time-based variables based on the `time_speed_up_scale` parameter, which indicates how fast the
      simulation is running in relation to real time.
    - Even if the acceleration function is turned on in Unity3D car simulation, the timestamp is still the same as the
      computer time (no acceleration).

---

## Modifying Observation Space and Action Space

If you wish to modify the observation space or action space in `CarRLEnvironment`, follow the steps below:

### Modifying the Observation Space

The observation space defines the structure of the data that the reinforcement learning (RL) model receives at each time
step. It is set in the environment's initialization and can be changed to include more or different data types from the
car.

For example, if you want to change the observation space to include RGB images instead of grayscale, or add more
telemetry data, such as acceleration or angular velocity, here’s what you can do:

1. **Modify the Observation Space**:
    - In the `__init__` method of `CarRLEnvironment`, update the observation space dictionary:
    ```python
    self.observation_space = spaces.Dict({
        "image": spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),  # Use (64, 64, 3) for RGB images
        "steering_speed": spaces.Box(low=np.array([-25.0, 0.0]), high=np.array([25.0, 100.0]), dtype=np.float32),
        "acceleration": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),  # Add acceleration (x, y, z)
        "angular_velocity": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)  # Add angular velocity (x, y, z)
    })
    ```

2. **Update the Observation Data**:
    - In the `step` and `reset` methods, update the `current_observation` dictionary to include the new data
      from `car_data`:
    ```python
    self.current_observation = {
        "image": processed_image,  # Process the RGB image
        "steering_speed": np.array([current_steering, current_speed], dtype=np.float32),
        "acceleration": np.array([car_data.acceleration_x, car_data.acceleration_y, car_data.acceleration_z]),
        "angular_velocity": np.array([car_data.angular_velocity_x, car_data.angular_velocity_y, car_data.angular_velocity_z])
    }
    ```

### Modifying the Action Space (Discrete Control)

The action space defines the set of possible actions that the RL agent can take at each step. By default, the action
space controls the car’s steering angle and throttle using continuous values. You can modify the action space to use
discrete control, which may be easier for certain algorithms or scenarios.

1. **Modify the Action Space**:
    - In the `__init__` method, change the action space to discrete values for steering and throttle. For example, if
      you want to define 3 discrete steering positions (left, straight, right) and 3 throttle positions (reverse,
      neutral, forward):
    ```python
    self.action_space = spaces.MultiDiscrete([3, 3])  # 3 discrete positions for steering and 3 for throttle
    ```

   In this case, the agent will choose actions as pairs of integers, where:
    - Steering: 0 = left, 1 = straight, 2 = right
    - Throttle: 0 = reverse, 1 = neutral, 2 = forward

2. **Update the Action Logic**:
    - In the `step` method, map the discrete action values to actual steering and throttle values. For example:
    ```python
    steering, throttle = action
    if steering == 0:
        steering_angle = -1.0  # Full left
    elif steering == 1:
        steering_angle = 0.0   # Straight
    else:
        steering_angle = 1.0   # Full right

    if throttle == 0:
        throttle_value = -1.0  # Reverse
    elif throttle == 1:
        throttle_value = 0.0   # Neutral
    else:
        throttle_value = 1.0   # Forward

    # Send the control to the car service
    self.car_service.send_control(steering_angle, throttle_value)
    ```

### Important Notes:

- When using discrete actions, be sure to appropriately map the discrete values to meaningful continuous values in the
  action space.
- You can expand the `MultiDiscrete` action space to include more combinations (e.g., adding brake control as another
  discrete value).
- This approach simplifies the action selection process, especially for algorithms that work better with discrete action
  spaces.

By switching to discrete actions, you can create more structured control strategies, which may be beneficial in certain
scenarios, such as grid-based actions or pre-defined control steps.
