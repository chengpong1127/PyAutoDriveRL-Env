import os

import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import torch.nn as nn
from gymnasium import spaces

from CarRLEnvironment import CarRLEnvironment
from CarDataService import CarSocketService

import wandb
from wandb.integration.sb3 import WandbCallback

from model import ImageEncoder, ImageEncoderSWIN

wandb.init(
    project="RL-Final",
)


class CustomCNN(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor for handling image input and extracting features.

    Args:
        observation_space (spaces.Dict): The observation space which includes the image input.
        features_dim (int): The dimension of the output feature vector after CNN layers.
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        # Extract the 'image' shape from observation space, assuming image is (64, 64, 3)
        super(CustomCNN, self).__init__(observation_space, features_dim)
        image_feature_dim = 20
        self.image_size = (60, 120)
        #self.image_encoder = ImageEncoderSWIN(observation_space['image'].shape, image_feature_dim)
        self.image_encoder = ImageEncoder((5, *self.image_size), image_feature_dim)

        # Define a fully connected layer to combine CNN output with other inputs (steering/speed)
        self.mlp = nn.Sequential(
            nn.Linear(image_feature_dim + 19, features_dim),  # Add steering and speed (2,)
            nn.ReLU(),
        )

    def forward(self, observations):
        """
        Forward pass for feature extraction.

        Args:
            observations (dict): A dictionary containing 'image' and 'steering_speed' inputs.

        Returns:
            Tensor: A tensor representing extracted features from image and steering/speed.
        """
        cat_features = ['steering_angle', 'throttle', 'speed', 'velocity', 'acceleration', 'angular_velocity', 'wheel_friction', 'orientation', 'brake_input']
        
        image = observations['image']
        line_image = observations['line_image']
        depth_image = observations['depth_image']
        hybrid_image = th.cat([image, line_image, depth_image], dim=1)
        hybrid_image = th.nn.functional.interpolate(hybrid_image, size=self.image_size, mode='bilinear')
        
        hybrid_feature = self.image_encoder(hybrid_image)

        total_features = th.cat([hybrid_feature] + [observations[cat_feature] for cat_feature in cat_features], dim=1)
        # concat with obstacle_car
        total_features = th.cat([total_features, observations['obstacle_car'].to(th.float32)], dim=1)
        return self.mlp(total_features)


if __name__ == '__main__':
    """
    Main training loop for the car RL environment using SAC (or you can change it to PPO).

    Modifiable sections are marked with comments to help first-time users easily adjust the code.
    """

    # Initialize the CarSocketService
    car_service = CarSocketService(system_delay=0.1)  # Modify system delay to match the environment

    # Initialize the custom RL environment
    env = CarRLEnvironment(car_service)  # Adjust frame_stack_num based on how many frames to stack

    # Check if the environment follows the Gym standards
    check_env(env)

    # Define policy arguments with the custom CNN feature extractor
    policy_kwargs = {
        "features_extractor_class": CustomCNN,
        "features_extractor_kwargs": {"features_dim": 64},  # Change feature dimensions if needed
    }

    # Choose between SAC or PPO model (PPO used here for example)
    # model = SAC("MultiInputPolicy", env, policy_kwargs=policy_kwargs, buffer_size=2048, verbose=1, batch_size=64)
    model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1, n_steps=128, batch_size=64, n_epochs=10, learning_rate=0.003)
    # if os.path.exists(f"{model.__class__.__name__}_best_model.zip"):
    #     print("loading model...")
    #     model.load(f"{model.__class__.__name__}_best_model.zip")

    # Set training parameters
    total_timesteps = 2000  # Number of timesteps to train in each loop
    save_interval = 1000  # How often to save the model (in timesteps)
    best_reward = -np.inf  # Initial best reward
    best_model_path = f"{model.__class__.__name__}_best_model"  # Path to save the best model
    latest_model_path = f"{model.__class__.__name__}_latest_model"  # Path to save the latest model

    print(f"Training {model.__class__.__name__} model...")

    # Initialize observation and info
    obs, info = env.reset()

    # Training loop
    while True:
        """
        This loop will continuously train the model and evaluate its performance.
        Modify the total_timesteps and save_interval to adjust the training frequency and model saving.
        """

        # Train the model for a specified number of timesteps
        model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False, callback=WandbCallback())

        # Save the latest model after each training step
        print(f"Saving latest model: {latest_model_path}")
        model.save(latest_model_path)

        # Evaluate the model by running an episode and calculate the total reward
        total_reward = 0
        done = False
        while not done:
            # Use the model to predict the next action
            action, _states = model.predict(obs)

            # Execute the action in the environment and get feedback
            obs, rewards, done, truncated, info = env.step(action)

            # Accumulate the reward for this episode
            total_reward += rewards

            # If the episode is done, reset the environment
            if done:
                obs, info = env.reset()

        # Check if this is the best model so far, and save it
        if total_reward > best_reward:
            best_reward = total_reward
            model.save(best_model_path)  # Save the model with the highest reward
            print(f"New best model saved with total reward: {total_reward}")

        # Print training progress
        print(f"Training step complete. Latest model saved. Total reward this episode: {total_reward}")

