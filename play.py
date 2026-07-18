"""
play.py - Load Trained DQN Agent and Play Atari Game
Uses GreedyQPolicy for deterministic action selection.
"""

import time
import numpy as np
import cv2
import ale_py
import gymnasium as gym
gym.register_envs(ale_py)

from stable_baselines3 import DQN


# ============================================================================
# REUSE WRAPPERS FROM train.py
# ============================================================================

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        truncated = False
        for _ in range(self._skip):
            obs, reward, done, truncated, info = self.env.step(action)
            total_reward += reward
            if done or truncated:
                break
        return obs, total_reward, done, truncated, info


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84, grayscale=True):
        super().__init__(env)
        self.width = width
        self.height = height
        self.grayscale = grayscale
        if grayscale:
            self.observation_space = gym.spaces.Box(
                low=0, high=255, shape=(height, width, 1), dtype=np.uint8
            )
        else:
            self.observation_space = gym.spaces.Box(
                low=0, high=255, shape=(height, width, 3), dtype=np.uint8
            )

    def observation(self, frame):
        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        if self.grayscale:
            frame = np.expand_dims(frame, -1)
        return frame


class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reward(self, reward):
        return np.sign(reward)


class FrameStack(gym.Wrapper):
    def __init__(self, env, n_frames=4):
        super().__init__(env)
        self.n_frames = n_frames
        self.frames = []
        obs_space = env.observation_space
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(n_frames, obs_space.shape[0], obs_space.shape[1]),
            dtype=np.uint8
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frames = [obs] * self.n_frames
        return self._get_obs(), info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self.frames.append(obs)
        self.frames.pop(0)
        return self._get_obs(), reward, done, truncated, info

    def _get_obs(self):
        return np.concatenate(self.frames, axis=2).transpose(2, 0, 1).astype(np.uint8)


# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

def get_env_id():
    """Get the correct environment name."""
    env_names = [
        "ALE/Pong-v5",
        "ALE/Breakout-v5",
        "ALE/SpaceInvaders-v5",
    ]
    for env_id in env_names:
        try:
            env = gym.make(env_id)
            env.close()
            print(f"✓ Using environment: {env_id}")
            return env_id
        except Exception:
            continue
    raise ValueError("No Atari environment found!")


def create_play_env(env_id, render_mode="human"):
    """Create environment for playing with rendering."""
    env = gym.make(env_id, render_mode=render_mode)
    env = MaxAndSkipEnv(env, skip=4)
    env = WarpFrame(env, width=84, height=84, grayscale=True)
    env = ClipRewardEnv(env)
    env = FrameStack(env, n_frames=4)
    return env


def play_game(env_id, model_path="dqn_model", num_episodes=5):
    """Load the trained model and play the game."""
    
    print("\n" + "="*70)
    print("LOADING TRAINED DQN AGENT")
    print("="*70)
    
    try:
        model = DQN.load(model_path)
        print(f"✓ Model loaded from: {model_path}.zip")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("  Make sure you have trained the model first using train.py")
        return
    
    env = create_play_env(env_id, render_mode="human")
    print(f"✓ Environment: {env_id}")
    
    print("\n" + "="*70)
    print("AGENT PLAYING (GreedyQPolicy)")
    print("="*70)
    print("The agent selects actions with the highest Q-value.")
    print("Close the game window to stop playing.\n")
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(1, num_episodes + 1):
        print(f"\nEpisode {episode}/{num_episodes}")
        print("-" * 50)
        
        obs, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0
        step_count = 0
        
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            if step_count % 100 == 0:
                print(f"  Step {step_count}: Reward={total_reward:.2f}")
            
            time.sleep(0.005)
        
        episode_rewards.append(total_reward)
        episode_lengths.append(step_count)
        
        print(f"\nEpisode {episode} Complete:")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Steps: {step_count}")
    
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    print(f"Mean Reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")
    print(f"Mean Length: {np.mean(episode_lengths):.2f} steps")
    print(f"Best Episode: {max(episode_rewards):.2f}")
    
    env.close()


if __name__ == "__main__":
    ENV_ID = get_env_id()
    play_game(
        env_id=ENV_ID,
        model_path="dqn_model",
        num_episodes=5
    )