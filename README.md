# 🏎️ Autonomous Racing using Reinforcement Learning

A reinforcement learning project focused on training an autonomous racing agent to navigate a custom gate-based racing environment using the **Soft Actor-Critic (SAC)** algorithm.

The project explores continuous vehicle control, autonomous navigation, and learning progression through simulation.

## 🎥 Training Progression

Watch how the agent's behaviour evolves across different training episodes.

| Episode 10 | Episode 40 |
|:---:|:---:|
| ![Episode 10](animations_restful/episode_10.gif) | ![Episode 40](animations_restful/episode_40.gif) |
| **Initial Training** | **Early Learning** |

| Episode 100 | Episode 400 |
|:---:|:---:|
| ![Episode 100](animations_restful/episode_100.gif) | ![Episode 400](animations_restful/episode_400.gif) |
| **Intermediate Training** | **Later Training** |

*The animations show the agent at different training checkpoints. Compare its navigation behaviour and vehicle control across episodes.*

> Replace the GIF filenames above with your actual filenames if they differ. GitHub README files display GIF animations directly. For MP4 videos, use linked video files or GitHub video attachments instead of Markdown image syntax.

## 📌 Project Overview

This project implements a reinforcement learning agent that learns to navigate a racing track by interacting with a custom simulation environment.

The agent uses the Soft Actor-Critic algorithm, a model-free off-policy reinforcement learning algorithm designed for continuous action spaces.

### Key Features

- Custom gate-based racing environment.
- Autonomous vehicle navigation.
- Continuous action space for velocity and rotation.
- SAC-based reinforcement learning.
- Vehicle dynamics with acceleration limits.
- Training checkpoints and simulation animations.
- Saved trained models and training logs.

## 🧠 Algorithm

**Soft Actor-Critic (SAC)**

SAC is an off-policy actor-critic reinforcement learning algorithm that optimises expected cumulative rewards while encouraging exploration through entropy regularisation.

Implementation: [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| PyTorch | Deep learning framework |
| Stable-Baselines3 | SAC implementation |
| Gymnasium | Custom simulation environment |
| NumPy | Numerical computations |
| Matplotlib | Training visualisation |
| FFmpeg | Animation generation |

## 📂 Project Structure

```text
Racing-Environment/
│
├── agent.py
├── race_acc.py
├── requirements.txt
│
├── animations_restful/
│   ├── episode_10.gif
│   ├── episode_40.gif
│   ├── episode_100.gif
│   └── episode_400.gif
│
├── Custom_Environment/
│   └── training_results.txt
│
└── trained_models/
    └── model.zip
```

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone YOUR_REPOSITORY_URL
cd Racing-Environment
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Install FFmpeg

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Linux:**
```bash
sudo apt-get install ffmpeg
```

**Windows:**

Download FFmpeg from https://ffmpeg.org/download.html

## 🚀 Running the Project

Start training the autonomous racing agent:

```bash
python agent.py
```

The training script:
- Initialises the SAC agent.
- Interacts with the racing environment.
- Learns continuous vehicle control.
- Saves simulation animations at configured checkpoints.
- Saves the trained model and training logs.

## 📊 Training Evaluation

The project includes visual checkpoints at:

- Episode 10
- Episode 40
- Episode 100
- Episode 400

These checkpoints provide a visual comparison of the agent's behaviour during training.

Further quantitative evaluation can include:
- Average episode reward.
- Number of gates successfully passed.
- Episode completion rate.
- Navigation consistency.
- Vehicle control stability.

## 🔬 Future Improvements

- Evaluate navigation performance across multiple random seeds.
- Analyse reward progression during training.
- Improve exploration and reward design.
- Investigate training stability and convergence.
- Compare SAC with other reinforcement learning algorithms.
- Evaluate generalisation to different track configurations.

## 👩‍💻 Contributor

**Arpitha Thimmegowda**

[![GitHub](https://img.shields.io/badge/GitHub-arpitha--co-black?logo=github)](https://github.com/arpitha-co)

---

*Exploring autonomous systems through reinforcement learning.*
