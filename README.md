# PyAutoDriveRL-Env

**PyAutoDriveRL-Env** is a Python-based reinforcement learning (RL) framework designed for training and simulating autonomous driving in a Unity3D environment. The project integrates RL techniques using libraries like Stable Baselines3, and leverages Unity3D for realistic car simulation environments.

## Features
- **Reinforcement Learning**: Uses algorithms like PPO and SAC to train autonomous driving models.
- **Unity3D Integration**: Communicates with Unity3D to simulate car environments.
- **Custom CNN Feature Extractor**: For handling camera input from the car simulation.
- **Car Simulation Control**: Provides throttle, steering, and reset triggers based on RL models.

## Additional Information
- I will provide a Unity3D executable, and in future releases, I will also provide the Unity project files to allow for custom environment editing.
- This project provides an environment wrapped using the Gym interface, allowing users to train models with Stable Baselines.
- A [Stable Baselines training example script](train_stable_baseline.py) is provided to help you get started quickly.
- A more flexible [training template script](train_my.py) is also available for users who want to modify the training logic or customize the environment further.

## File Structure
- **`CarDataService.py`**: Handles communication between the car simulation and Python. Manages car telemetry, including speed, position, and camera images.
- **`CarRLEnvironment.py`**: Defines the RL environment compatible with Gym, using car data for observations and rewards.
- **`train_stable_baseline.py`**: Main training script using Stable Baselines3 (PPO/SAC) for RL training.
- **`train_my.py`**: Custom RL training loop with reward computation and action selection.
- **`inference_template.py`**: Example script for performing inference using the trained RL model.

## Getting Started
1. **Requirements**:
    - Python 3.7+
    - Unity3D Simulator
    - Libraries: `numpy`, `opencv-python`, `stable-baselines3`, `gymnasium`, `torch`

2. **Install dependencies**:
    ```bash
    pip install numpy opencv-python stable-baselines3 gymnasium torch
    ```

3. **Run Training**:
    ```bash
    python train_stable_baseline.py
    ```

4. **Inference**:
    After training, you can perform inference using:
    ```bash
    python inference_template.py
    ```

## Customization
- Modify the **`CarRLEnvironment.py`** to adjust observation space or reward functions.
- Implement your custom RL algorithms in **`train_my.py`** if you prefer not to use Stable Baselines3.

## Enviroment Observation
[ObservationDetail.md](doc/ObservationDetail.md)

## Gym-like enviroment detail
[Gym-likeEnviromentDetail.md](doc/Gym-likeEnviromentDetail.md)

## Acknowledgments
This project is built using:
- **Stable Baselines3** for RL algorithms.
- **Unity3D** for car simulation.
