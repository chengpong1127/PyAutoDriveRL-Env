# PyAutoDriveRL-Env

**PyAutoDriveRL-Env** is a Python-based reinforcement learning (RL) framework designed for training and simulating
autonomous driving in a Unity3D environment. The project integrates RL techniques using libraries like Stable
Baselines3, and leverages Unity3D for realistic car simulation environments.

https://github.com/user-attachments/assets/e4b72665-a27d-40d5-8042-57feab643eef

![SystemFramework.png](doc%2FSystemFramework.png)

## 🚗 Unity3D Car Simulation Environment

- Made using Unity3D, the project will be made public in the future for everyone to build own car simulation environment.
- [Windows download link](https://gofile.me/7jNiV/q7CELHz77), [Linux download link](https://gofile.me/7jNiV/4fe30vS9P)(Contains startup documentation)
- Double click the `Car.exe` to start the car simulation environment
- When starting the Unity simulation environment, you can manually control the car’s movement using W, A, S, and D keys for forward, backward, and directional movement.
- Control priority is given first to keyboard control, followed by Python control.

## 📑 Update
- 2024/11/18: Fix: Fixed a bug that would cause progress to be unable to be traced if a different location was specified for reset. The reason is that as long as it is reset, the progress tracker will be reset at the starting point.
- 2024/11/09: Fix: Bug caused by other scripts not keeping up with the new usage of reset_trigger in the 11/08 update
- 2024/11/08: New features: When resetting the vehicle's position, you can now specify the checkpoint from which to restart. Please refer to the [CarRLEnvironmentControlGuide.md](doc%2FCarRLEnvironmentControlGuide.md) for details

## 🔍 Additional Information

- Unity3D project will be provided, and in future releases, the Unity3D project to allow for custom environment
  editing will also be provided.
- This project features an environment wrapped using the [Gym interface](CarRLEnvironment.py), allowing users to train models with Stable
  Baselines.
- A [Stable Baselines training example script](train_stable_baseline.py) is provided to help you get started quickly.
- A more flexible [training template script](train_my.py) is also available for users who want to modify the training
  logic or customize the environment further.

## ❗❗❗❗ Notice ❗❗❗❗

- If you plan to use speed-up functionality during training, make sure your model maintains consistent stability across
  different FPS levels (high/low).
- For instance, if you use information from the previous and next frame as input to the model, inconsistencies in the
  time interval between frames may lead to unexpected issues.

## 🏬 File Structure

- **`CarDataService.py`**: Handles communication between the car simulation and Python. Manages car telemetry, including
  speed, position, and camera images.
- **`CarRLEnvironment.py`**: Defines the GYM-compatible RL environment, using car data for observations and rewards.
- **`train_stable_baseline.py`**: Main training script using Stable Baselines3 (PPO/SAC) for RL training.
- **`train_my.py`**: Custom RL training loop with reward computation and action selection.
- **`inference_template.py`**: Example script for performing inference using the trained RL model.

## 📒 Getting Started

1. **Requirements**:
    - OS: `windows`
    - Python version: 3.10
    - Libraries: `numpy==1.26.3`, `opencv-python==4.6.0.66`, `stable-baselines3==2.3.2`, `gymnasium==0.29.1`, `torch==2.1.2`,`ultralytics==8.3.27`,`transformers==4.36.2`

2. **Install dependencies**:
    ```bash
    conda create -n autodrive_rl python=3.10
    conda activate autodrive_rl
    pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
    pip install gymnasium==0.29.1
    pip install stable-baselines3==2.3.2
    pip install opencv-python==4.6.0.66
    pip install "python-socketio<4.3" "python-engineio<3.9"
    pip install eventlet
    pip install flask
    pip install ultralytics==8.3.27
    pip install transformers
    ```
    * CPU version
    ```bash
    conda create -n autodrive_rl python=3.10
    conda activate autodrive_rl
    pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cpu
    pip install gymnasium==0.29.1
    pip install stable-baselines3==2.3.2
    pip install opencv-python==4.6.0.66
    pip install "python-socketio<4.3" "python-engineio<3.9"
    pip install eventlet
    pip install flask
    pip install ultralytics==8.3.27
    pip install transformers
    ```

3. **Run Training** (Please start the **_Unity3D Car Simulation Environment_** first):
    ```bash
    conda activate autodrive_rl
    python train_stable_baseline.py
    ```

4. **Inference** (Please start the **_Unity3D Car Simulation Environment_** first):
    ```bash
    conda activate autodrive_rl
    python inference_template.py
    ```
   
    After stablebaseline training, you can perform inference using:
    ```bash
    conda activate autodrive_rl
    python inference_stablebaseline_template.py
    ```

## 🛠️ Customization

- Modify the **`CarRLEnvironment.py`** to adjust observation space or reward functions.
- Implement your custom RL algorithms in **`train_my.py`** if you prefer not to use Stable Baselines3.

## 📜 Environment Observation Detail

[ObservationDetail.md](doc/ObservationDetail.md)

## 📜 How to control the Car Simulation Environment / Environment Action Space Detail

[CarRLEnvironmentControlGuide.md](doc/CarRLEnvironmentControlGuide.md)

## 📜 Gym-like environment detail

[Gym-likeEnvironmentDetail.md](doc/Gym-likeEnvironmentDetail.md)

## ❤️ Acknowledgments

This project is built using:

- **Stable Baselines3** for RL algorithms.
- **Unity 3D car simulation environment** modification
  from [GitHub repo](https://github.com/udacity/self-driving-car-sim.git). 
