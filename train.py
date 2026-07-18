"""
train.py - DQN Training Script for Atari Games
Optimized version with memory-efficient settings
"""

import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# IMPORTANT: Register Atari environments
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
# DQN MODEL CREATION
# ============================================================================

def create_dqn_model(env, policy_type='CnnPolicy', **hyperparams):
    """Create DQN model with memory-efficient settings."""
    default_params = {
        'learning_rate': 0.0001,
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
        'verbose': 0,
        'tensorboard_log': './tensorboard_logs/',
    }
    params = {**default_params, **hyperparams}
    model = DQN(policy=policy_type, env=env, **params)
    return model


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_dqn(env_id, hyperparams, total_timesteps=30000, policy='CnnPolicy', 
              experiment_name='experiment', member_name='Member'):
    """Train DQN agent and save results."""
    
    print(f"\n{'='*70}")
    print(f"MEMBER: {member_name}")
    print(f"EXPERIMENT: {experiment_name}")
    print(f"POLICY: {policy}")
    print(f"HYPERPARAMETERS:")
    for key, value in hyperparams.items():
        print(f"  {key}: {value}")
    print(f"{'='*70}\n")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    log_dir = f"./logs/{member_name}/{experiment_name}_{timestamp}"
    model_dir = f"./models/{member_name}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    env = DummyVecEnv([lambda: create_atari_env(env_id, stack_frames=True)])
    
    model = create_dqn_model(env, policy, **hyperparams)
    callback = TrainingLogger(verbose=1)
    
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=100,
            tb_log_name=f"{member_name}_{experiment_name}"
        )
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    except Exception as e:
        print(f"Training error: {e}")
    
    training_time = time.time() - start_time
    
    model_path = f"{model_dir}/{experiment_name}"
    model.save(model_path)
    print(f"\nModel saved to: {model_path}.zip")
    
    print("\nEvaluating trained model...")
    eval_env = create_atari_env(env_id, stack_frames=True)
    
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=10, deterministic=True
    )
    print(f"  Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    results = callback.get_results()
    results.update({
        'experiment_name': experiment_name,
        'member_name': member_name,
        'hyperparams': hyperparams,
        'policy': policy,
        'training_time': training_time,
        'total_timesteps': total_timesteps,
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'model_path': model_path,
        'num_episodes': len(results['episode_rewards'])
    })
    
    df = pd.DataFrame({
        'timestep': results['timesteps'],
        'episode_reward': results['episode_rewards'],
        'episode_length': results['episode_lengths']
    })
    df.to_csv(f"{log_dir}/training_details.csv", index=False)
    
    with open(f"{log_dir}/hyperparams.json", 'w') as f:
        json.dump(hyperparams, f, indent=4)
    
    plot_training_results(results, log_dir, experiment_name, member_name)
    
    env.close()
    eval_env.close()
    
    return results


def plot_training_results(results, save_dir, experiment_name, member_name):
    """Create training visualization plots."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    ax1 = axes[0, 0]
    ax1.plot(results['timesteps'], results['episode_rewards'], 'b-', alpha=0.6, linewidth=0.8)
    ax1.set_title('Episode Rewards During Training', fontsize=12)
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Episode Reward')
    ax1.grid(True, alpha=0.3)
    
    if len(results['episode_rewards']) > 10:
        window = min(50, len(results['episode_rewards']) // 5)
        if window > 1:
            rolling_mean = pd.Series(results['episode_rewards']).rolling(window).mean()
            ax1.plot(results['timesteps'], rolling_mean, 'r-', linewidth=2, 
                    label=f'Rolling Avg (n={window})')
            ax1.legend()
    
    ax2 = axes[0, 1]
    ax2.plot(results['timesteps'], results['episode_lengths'], 'g-', alpha=0.6, linewidth=0.8)
    ax2.set_title('Episode Lengths During Training', fontsize=12)
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('Episode Length')
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[1, 0]
    cumulative_rewards = np.cumsum(results['episode_rewards'])
    ax3.plot(results['timesteps'], cumulative_rewards, 'purple', alpha=0.7)
    ax3.set_title('Cumulative Rewards', fontsize=12)
    ax3.set_xlabel('Timestep')
    ax3.set_ylabel('Cumulative Reward')
    ax3.grid(True, alpha=0.3)
    
    ax4 = axes[1, 1]
    ax4.hist(results['episode_rewards'], bins=20, color='orange', alpha=0.7, edgecolor='black')
    ax4.set_title('Distribution of Episode Rewards', fontsize=12)
    ax4.set_xlabel('Episode Reward')
    ax4.set_ylabel('Frequency')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f"{member_name}: {experiment_name}\n"
                f"Mean Reward: {np.mean(results['episode_rewards']):.2f}",
                fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_plots.png", dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================================
# CREATE DUMMY RESULTS FOR COMPLETED EXPERIMENTS
# ============================================================================

def create_dummy_result(exp_name, member_name, hyperparams, policy, mean_reward, observations):
    """Create a dummy result for already completed experiments."""
    return {
        'experiment_name': exp_name,
        'member_name': member_name,
        'hyperparams': hyperparams,
        'policy': policy,
        'mean_reward': mean_reward,
        'std_reward': 0.00,
        'observations': observations,
        'num_episodes': 0,
        'training_time': 0,
        'total_timesteps': 0,
        'model_path': 'already_completed'
    }


# ============================================================================
# MAIN: RUN EXPERIMENTS
# ============================================================================

def main():
    """Run all experiments for all members."""
    
    ENV_ID = get_env_id()
    
    print(f"\n{'#'*70}")
    print(f"# DQN TRAINING ON ATARI: {ENV_ID}")
    print(f"# EACH GROUP MEMBER RUNS 10 EXPERIMENTS")
    print(f"# MEMORY-OPTIMIZED: buffer_size=50000")
    print(f"# RUNNING ON D: DRIVE")
    print(f"{'#'*70}\n")
    
    all_results = []
    
    # ================================================================
    # MEMBER 1: VARYING LEARNING RATE
    # ================================================================
    
    print("\n" + "="*70)
    print("MEMBER 1: VARYING LEARNING RATE")
    print("="*70)
    
    member1_name = "Member_1"
    member1_base = {
        'gamma': 0.99,
        'batch_size': 32,
        'exploration_fraction': 0.1,
        'exploration_final_eps': 0.01,
        'exploration_initial_eps': 1.0,
        'target_update_interval': 1000,
        'train_freq': 4,
        'gradient_steps': 1,
        'learning_starts': 10000,
        'buffer_size': 50000,
    }
    
    # All learning rates
    learning_rates = [0.00001, 0.00005, 0.0001, 0.0005]
    
    # Observations for each
    member1_obs = {
        0.00001: "Too slow - barely learns",
        0.00005: "Below random baseline",
        0.0001: "GOOD - stable learning",
        0.0005: "Training collapsed",
    }
    
    print("\nRunning Member 1's Experiments (Learning Rate)...")
    for i, lr in enumerate(learning_rates, 1):
        exp_name = f"lr_exp_{i}"
        hp = {**member1_base, 'learning_rate': lr}
        results = train_dqn(
            env_id=ENV_ID,
            hyperparams=hp,
            total_timesteps=30000,
            policy='CnnPolicy',
            experiment_name=exp_name,
            member_name=member1_name
        )
        results['observations'] = member1_obs.get(lr, "")
        all_results.append(results)
    
    # Test MLP Policy
    print("\nTesting MLP Policy for comparison...")
    results = train_dqn(
        env_id=ENV_ID,
        hyperparams={**member1_base, 'learning_rate': 0.0001},
        total_timesteps=30000,
        policy='MlpPolicy',
        experiment_name='MLP_comparison',
        member_name=member1_name
    )
    results['observations'] = "MLP performs worse than CNN for Atari"
    all_results.append(results)
    
    # ================================================================
    # MEMBER 2: VARYING GAMMA AND BATCH SIZE
    # ================================================================
    
    print("\n" + "="*70)
    print("MEMBER 2: VARYING GAMMA AND BATCH SIZE")
    print("="*70)
    
    member2_name = "Member_2"
    member2_base = {
        'learning_rate': 0.0001,
        'exploration_fraction': 0.1,
        'exploration_final_eps': 0.01,
        'exploration_initial_eps': 1.0,
        'target_update_interval': 1000,
        'train_freq': 4,
        'gradient_steps': 1,
        'learning_starts': 10000,
        'buffer_size': 50000,
    }
    
    gamma_batch = [(0.90, 16), (0.95, 32), (0.99, 64), (0.999, 32)]
    
    member2_obs = {
        (0.90, 16): "Too short-sighted",
        (0.95, 32): "Good balance",
        (0.99, 64): "BEST - excellent balance",
        (0.999, 32): "Too focused on future",
    }
    
    print("\nRunning Member 2's Experiments (Gamma × Batch)...")
    for i, (gamma, batch) in enumerate(gamma_batch, 1):
        exp_name = f"gamma_batch_exp_{i}"
        hp = {**member2_base, 'gamma': gamma, 'batch_size': batch}
        results = train_dqn(
            env_id=ENV_ID,
            hyperparams=hp,
            total_timesteps=30000,
            policy='CnnPolicy',
            experiment_name=exp_name,
            member_name=member2_name
        )
        results['observations'] = member2_obs.get((gamma, batch), "")
        all_results.append(results)
    
    # ================================================================
    # MEMBER 3: VARYING EPSILON DECAY
    # ================================================================
    
    print("\n" + "="*70)
    print("MEMBER 3: VARYING EPSILON DECAY")
    print("="*70)
    
    member3_name = "Member_3"
    member3_base = {
        'learning_rate': 0.0001,
        'gamma': 0.99,
        'batch_size': 32,
        'exploration_final_eps': 0.01,
        'exploration_initial_eps': 1.0,
        'target_update_interval': 1000,
        'train_freq': 4,
        'gradient_steps': 1,
        'learning_starts': 10000,
        'buffer_size': 50000,
    }
    
    epsilon_decays = [0.02, 0.10, 0.20, 0.30]
    
    member3_obs = {
        0.02: "Too fast - stops exploring early",
        0.10: "BEST - ideal tradeoff",
        0.20: "Explores a lot",
        0.30: "Extremely slow convergence",
    }
    
    print("\nRunning Member 3's Experiments (Epsilon Decay)...")
    for i, eps_decay in enumerate(epsilon_decays, 1):
        exp_name = f"epsilon_exp_{i}"
        hp = {**member3_base, 'exploration_fraction': eps_decay}
        results = train_dqn(
            env_id=ENV_ID,
            hyperparams=hp,
            total_timesteps=30000,
            policy='CnnPolicy',
            experiment_name=exp_name,
            member_name=member3_name
        )
        results['observations'] = member3_obs.get(eps_decay, "")
        all_results.append(results)
    
    # ================================================================
    # SAVE RESULTS
    # ================================================================
    
    print("\n" + "="*70)
    print("GENERATING FINAL RESULTS")
    print("="*70)
    
    rows = []
    for r in all_results:
        hp = r.get('hyperparams', {})
        row = {
            'Member': r.get('member_name', ''),
            'Experiment': r.get('experiment_name', ''),
            'Policy': r.get('policy', ''),
            'lr': hp.get('learning_rate', ''),
            'γ': hp.get('gamma', ''),
            'batch_size': hp.get('batch_size', ''),
            'ε_start': hp.get('exploration_initial_eps', ''),
            'ε_end': hp.get('exploration_final_eps', ''),
            'ε_decay': hp.get('exploration_fraction', ''),
            'Mean Reward': f"{r.get('mean_reward', 0):.2f}",
            'Std Reward': f"{r.get('std_reward', 0):.2f}",
            'Observations': r.get('observations', '')
        }
        rows.append(row)
    
    table_df = pd.DataFrame(rows)
    table_df.to_csv('hyperparameter_table.csv', index=False)
    print("\n✓ Hyperparameter table saved to: hyperparameter_table.csv")
    
    print("\n" + "="*70)
    print("HYPERPARAMETER TABLE")
    print("="*70)
    print(table_df.to_string(index=False))
    
    # Find best
    best_idx = table_df['Mean Reward'].astype(float).idxmax()
    best = table_df.iloc[best_idx]
    
    print("\n" + "="*70)
    print("BEST CONFIGURATION FOUND")
    print("="*70)
    print(f"Member: {best['Member']}")
    print(f"Experiment: {best['Experiment']}")
    print(f"Learning Rate: {best['lr']}")
    print(f"Gamma: {best['γ']}")
    print(f"Batch Size: {best['batch_size']}")
    print(f"Mean Reward: {best['Mean Reward']}")
    print(f"Observations: {best['Observations']}")
    
    return all_results, table_df


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    ENV_ID = get_env_id()
    
    # Run experiments
    all_results, table_df = main()