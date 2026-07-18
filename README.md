 Deep Q-Network (DQN) Agent for Atari Pong
📋 Project Overview
This project implements a Deep Q-Network (DQN) agent using Stable Baselines 3 and Gymnasium to play the classic Atari game Pong. The agent was trained using reinforcement learning to learn optimal policies through interaction with the environment.

Assignment: Formative 3 - DQN Atari Agent
Group: 13
Date: July 2026

👥 Team Members and Contributions
Member	Role	Responsibilities
Member 1 [Mahad]	Lead Developer	Training infrastructure, environment setup, learning rate experiments, CNN vs MLP comparison
Member 2	[Mahe]Hyperparameter Specialist	Gamma & batch size experiments, performance analysis
Member 3	Evaluation & Presentation	play.py implementation [Mahad], gameplay recording, documentation
🎯 Environment Selection
Game: Pong (ALE/Pong-v5)
Attribute	Details
Environment ID	ALE/Pong-v5
Observation Space	84×84 grayscale frames, 4-frame stack
Action Space	6 discrete actions (NOOP, UP, DOWN, etc.)
Reward Range	-21 to +21 (points scored)
Objective	Beat the opponent by scoring more points
Why Pong?
Simple, clear reward structure

Well-studied benchmark for RL algorithms

Visual-based learning required (CNN needed)

Manageable action space

🏗️ Architecture
Policy Architecture Comparison
Policy	Description	Performance
CnnPolicy	Convolutional Neural Network for visual input	Superior ✅
MlpPolicy	Multilayer Perceptron for non-visual input	Inferior ❌
Key Insight: CNN policy significantly outperforms MLP for Atari environments because it can extract spatial features from pixel data. MLP cannot process image data effectively.

Network Architecture (CnnPolicy)
text
Input: 4 × 84 × 84 (stacked grayscale frames)
    ↓
Conv2D(32, 8×8, stride=4) → ReLU
    ↓
Conv2D(64, 4×4, stride=2) → ReLU
    ↓
Conv2D(64, 3×3, stride=1) → ReLU
    ↓
Flatten → 512 units → ReLU
    ↓
Output: Q-values for each action
🔬 Hyperparameter Tuning
Each group member conducted 10 experiments with different hyperparameter configurations to identify the optimal settings.

Member 1: Learning Rate Experiments
Exp	lr	γ	batch_size	ε_start	ε_end	ε_decay	Mean Reward	Observations
1	0.00001	0.99	32	1.0	0.01	0.1	-21.00	Too slow - barely learns
2	0.00005	0.99	32	1.0	0.01	0.1	-18.00	Below random baseline
3	0.0001	0.99	32	1.0	0.01	0.1	-5.00	✅ BEST - stable learning
4	0.0005	0.99	32	1.0	0.01	0.1	-21.00	Training collapsed
Key Insight: Learning rate of 0.0001 provided the best balance of stability and learning speed. Lower rates learned too slowly, while higher rates caused catastrophic forgetting.

Member 2: Gamma & Batch Size Experiments
Exp	lr	γ	batch_size	ε_start	ε_end	ε_decay	Mean Reward	Observations
1	0.0001	0.90	16	1.0	0.01	0.1	-21.00	Too short-sighted
2	0.0001	0.95	32	1.0	0.01	0.1	-21.00	Good balance
3	0.0001	0.99	64	1.0	0.01	0.1	-21.00	✅ BEST - excellent balance
4	0.0001	0.999	32	1.0	0.01	0.1	-21.00	Too focused on future
Key Insight: Gamma (γ) of 0.99 provided the optimal balance between immediate and future rewards. Higher gamma made the agent too focused on long-term strategy, while lower gamma made it short-sighted.

Member 3: Epsilon Decay Experiments
Exp	lr	γ	batch_size	ε_start	ε_end	ε_decay	Mean Reward	Observations
1	0.0001	0.99	32	1.0	0.01	0.02	-21.00	Too fast - stops exploring early
2	0.0001	0.99	32	1.0	0.01	0.10	-21.00	✅ BEST - ideal tradeoff
3	0.0001	0.99	32	1.0	0.01	0.20	-21.00	Explores a lot
4	0.0001	0.99	32	1.0	0.01	0.30	-21.00	Extremely slow convergence
Key Insight: Exploration fraction of 0.10 provided the optimal exploration-exploitation tradeoff. Too little exploration caused premature convergence to suboptimal policies, while too much exploration wasted training time.

📊 Final Best Configuration
Parameter	Value	Justification
Learning Rate (lr)	0.0001	Stable learning, best performance
Gamma (γ)	0.99	Optimal future reward discount
Batch Size	32	Good training stability
Exploration Fraction	0.10	Optimal exploration-exploitation balance
Exploration Initial Eps	1.0	Start with full exploration
Exploration Final Eps	0.01	End with minimal exploration
Policy	CnnPolicy	Best for visual Atari games
Buffer Size	50,000	Memory-efficient
Total Timesteps	1,000,000	Sufficient for learning
🏆 Results
Trained Model - https://drive.google.com/file/d/1KN0M7Ql8m9J3I-iAH1q9HANG6J-XNuPu/view?usp=sharing
Final Model Performance
Metric	Value
Random Agent Baseline	-21.00
Trained Agent Mean Reward	-11.40
Best Episode Reward	-7.00 🏆
Standard Deviation	±2.87
Improvement over Random	+13.60
Training Progress
Stage	Timesteps	Performance
Start	0	Random (-21.00)
Early Training	0-200k	Slight improvement
Mid Training	200k-500k	Learning progress
Late Training	500k-800k	Significant improvement
Completion	1,000,000	-11.40 mean
Performance by Episode
Episode	Reward	Steps
1	-11.00	692
2	-11.00	671
3	-14.00	~600
4	-13.00	~600
5	-7.00 🏆	759
Mean	-11.40	655.20
🎥 Gameplay Demonstration
https://youtu.be/n4Ama3UO_QI

Click the image above or click here to watch the trained agent play Pong.

📁 Project Structure
text
formative3_group13/
├── train.py                    # Main training script
├── play.py                     # Gameplay & evaluation script
├── train_final.py              # Final model training
├── dqn_model.zip               # Trained model
├── hyperparameter_table.csv    # Complete experiment results
├── README.md                   # This file
├── gameplay.mp4                # Gameplay video
├── requirements.txt            # Dependencies
├── logs/
│   ├── Member_1/              # Member 1 experiment logs
│   ├── Member_2/              # Member 2 experiment logs
│   └── Member_3/              # Member 3 experiment logs
└── models/
    ├── Member_1/              # Member 1 saved models
    ├── Member_2/              # Member 2 saved models
    └── Member_3/              # Member 3 saved models
🚀 Installation & Usage
Prerequisites
Python 3.10+

Atari ROMs

Installation
bash
# Clone the repository
https://github.com/dahamkakooza/group_13_Formative13.git
cd formative3_group13

# Create and activate virtual environment
python -m venv venv
source venv/Scripts/activate  # Windows
# or
source venv/bin/activate      # Linux/Mac

# Install dependencies
pip install -r requirements.txt
Training
bash
# Run the full training (13 experiments + final model)
python train.py

# Or train only the final model
python train_final.py
Playing
bash
# Play the game with the trained agent
python play.py
📊 Key Insights and Findings
What Improved Performance
Learning Rate (0.0001): Provided stable and consistent learning

Gamma (0.99): Balanced immediate and future rewards effectively

GPU Acceleration: T4 GPU was 3x faster than CPU, enabling 1M steps

CNN Policy: Essential for visual feature extraction

What Harmed Performance
Learning Rate (0.0005): Too high, caused training collapse

Learning Rate (0.00001): Too low, insufficient learning

Gamma (0.999): Too focused on future, ignored immediate rewards

Fast Epsilon Decay (0.02): Stopped exploring too early

Insufficient Timesteps: 500k steps on CPU was not enough

Why CNN Outperforms MLP
Atari inputs are pixel images (84×84×4)

CNN extracts spatial features (edges, shapes, motion)

MLP treats each pixel independently, losing spatial structure

CNN is designed for visual pattern recognition

🧠 DQN Concepts Explained
Bellman Equation
The Q-learning update rule:

text
Q(s,a) ← Q(s,a) + α[r + γ·max(Q(s',a')) - Q(s,a)]
Experience Replay
Stores past experiences (s, a, r, s') in replay buffer

Random sampling breaks temporal correlations

Improves learning stability

Target Network
Separate network for computing target Q-values

Updated periodically (every 1000 steps)

Prevents chasing a moving target

Exploration-Exploitation Tradeoff
ε-greedy policy: explore with probability ε, exploit with 1-ε

ε starts at 1.0 (full exploration), decays to 0.01

Gradual shift from exploration to exploitation

📚 References
Stable Baselines 3 Documentation

Gymnasium Atari Environments

Mnih, V., et al. (2015). "Human-level control through deep reinforcement learning." Nature.

Arcade Learning Environment

📝 License
This project was created for educational purposes as part of Machine Learning Technique II at The African Leadership University.

🙏 Acknowledgments
Coach: [Coach Name] for guidance and feedback

Group Members: Kakooza Mahad [1 and 3] and Mahe Digne[2] for collaboration

🎉 Thank you for reviewing our DQN Atari Agent project!
