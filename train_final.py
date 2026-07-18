"""
train_proven.py - Train with Proven Pong Settings
This uses the same settings that work for Atari games
"""

import os
import time
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ================================================================
# ATARI ENVIRONMENT SETUP
# ================================================================
import ale_py
import gymnasium as gym
gym.register_envs(ale_py)

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy


# ============================================================================
# MEMORY-EFFICIENT ATARI WRAPPERS
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
        import cv2
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
    env_names = ["ALE/Pong-v5"]
    
    for env_id in env_names:
        try:
            env = gym.make(env_id)
            env.close()
            print(f"✓ Using environment: {env_id}")
            return env_id
        except Exception as e:
            print(f"✗ {env_id} failed: {e}")
            continue
    
    raise ValueError("No Atari environment found!")

def create_atari_env(env_id, render_mode=None, stack_frames=True):
    """Create and wrap Atari environment."""
    env = gym.make(env_id, render_mode=render_mode)
    env = MaxAndSkipEnv(env, skip=4)
    env = WarpFrame(env, width=84, height=84, grayscale=True)
    env = ClipRewardEnv(env)
    if stack_frames:
        env = FrameStack(env, n_frames=4)
    return env


# ============================================================================
# TRAINING LOGGER CALLBACK
# ============================================================================

class TrainingLogger(BaseCallback):
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.timesteps = []
        self.current_timestep = 0
        self.episode_count = 0
        self.best_mean_reward = -np.inf
        
    def _on_step(self):
        self.current_timestep += 1
        
        if 'ep_info_buffer' in self.locals and len(self.locals['ep_info_buffer']) > self.episode_count:
            ep_info = self.locals['ep_info_buffer'][-1]
            reward = ep_info['r']
            length = ep_info['l']
            
            self.episode_count += 1
            self.episode_rewards.append(reward)
            self.episode_lengths.append(length)
            self.timesteps.append(self.current_timestep)
            
            if len(self.episode_rewards) >= 10:
                mean_reward = np.mean(self.episode_rewards[-10:])
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
            
            if self.verbose > 0 and self.episode_count % 10 == 0:
                recent_mean = np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else 0
                print(f"Episode {self.episode_count}: Reward={reward:.2f}, Recent Mean={recent_mean:.2f}")
        
        return True
    
    def get_results(self):
        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'timesteps': self.timesteps,
            'best_mean_reward': self.best_mean_reward
        }


# ============================================================================
# MAIN: TRAIN WITH PROVEN SETTINGS
# ============================================================================

def train_proven_model():
    """Train with settings that are known to work for Pong."""
    
    ENV_ID = get_env_id()
    
    print("\n" + "="*70)
    print("TRAINING WITH PROVEN PONG SETTINGS")
    print("="*70)
    print("These settings are based on standard DQN for Atari:")
    print("  - Learning Rate: 0.0001")
    print("  - Buffer Size: 50,000")
    print("  - Batch Size: 32")
    print("  - Total Timesteps: 500,000 (increases chance of learning)")
    print("="*70)
    
    # Standard DQN settings for Atari
    hyperparams = {
        'learning_rate': 0.0001,           # Standard for Atari
        'gamma': 0.99,
        'batch_size': 32,
        'buffer_size': 50000,
        'exploration_fraction': 0.1,
        'exploration_final_eps': 0.01,
        'exploration_initial_eps': 1.0,
        'target_update_interval': 1000,
        'train_freq': 4,
        'gradient_steps': 1,
        'learning_starts': 10000,
        'verbose': 1,
        'tensorboard_log': './tensorboard_logs/',
    }
    
    # Create environment
    env = DummyVecEnv([lambda: create_atari_env(ENV_ID, stack_frames=True)])
    
    # Create model
    print("\nCreating DQN model...")
    model = DQN('CnnPolicy', env, **hyperparams)
    
    # Train with 500,000 timesteps
    total_timesteps = 500000
    print(f"\nTraining for {total_timesteps:,} timesteps...")
    print("This will take approximately 3-4 hours on CPU...")
    print("\n💡 Tip: Let it run overnight or while you're away")
    
    callback = TrainingLogger(verbose=1)
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=100
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving model...")
    except Exception as e:
        print(f"\nTraining error: {e}")
    
    training_time = time.time() - start_time
    
    # Save the model
    model.save("dqn_model")
    print(f"\n✓ Final model saved as: dqn_model.zip")
    print(f"✓ Training time: {training_time/60:.2f} minutes")
    
    # Evaluate
    print("\nEvaluating trained model...")
    eval_env = create_atari_env(ENV_ID, stack_frames=True)
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=10, deterministic=True
    )
    print(f"\nFinal Model Performance:")
    print(f"  Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    # Show results summary
    results = callback.get_results()
    if results['episode_rewards']:
        print(f"\nTraining Summary:")
        print(f"  Total Episodes: {len(results['episode_rewards'])}")
        print(f"  Best Episode Reward: {max(results['episode_rewards']):.2f}")
        print(f"  Last 10 Episodes Mean: {np.mean(results['episode_rewards'][-10:]):.2f}")
        print(f"  First 10 Episodes Mean: {np.mean(results['episode_rewards'][:10]):.2f}")
        print(f"  Improvement: {np.mean(results['episode_rewards'][-10:]) - np.mean(results['episode_rewards'][:10]):.2f}")
    
    env.close()
    eval_env.close()
    
    return model, results


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("DQN PONG TRAINING - PROVEN SETTINGS")
    print("="*70)
    print("\nNote: Pong typically needs 1-2 million timesteps to learn well.")
    print("We'll train for 500,000 timesteps which should show some learning.")
    print("="*70)
    
    model, results = train_proven_model()
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print("Run the game with: python play.py")
    print("="*70)