================================================================================
                    Deep Reinforcement Learning - Task 2
================================================================================

OVERVIEW
--------
This project contains reinforcement learning agents trained on two different 
environments:
1. Race Environment (race_acc.py) - A racing track with gates
2. Infinite Valley Environment (Custom_Environment/) - Custom terrain navigation

The agents use Soft Actor-Critic (SAC) algorithm from Stable-Baselines3.


REQUIREMENTS
------------
- Python 3.8 or higher
- numpy
- gymnasium
- stable-baselines3
- torch (PyTorch)
- matplotlib
- imageio-ffmpeg
- ffmpeg (system dependency - install via apt/yum/brew)


INSTALLATION
------------
1. Navigate to the project directory:
   cd "/Users/arpitha/Desktop/OTH/3rd Semester/DRL/Module_work/Task_2/Final"

2. Install Python dependencies:
   pip install -r requirements.txt

3. Install ffmpeg (for video/animation generation):
   - macOS: brew install ffmpeg
   - Linux: sudo apt-get install ffmpeg  OR  sudo yum install ffmpeg
   - Windows: Download from https://ffmpeg.org/download.html


RUNNING THE CODE
----------------

1. RACE ENVIRONMENT (Main Agent):
   ---------------------------------
   To train the agent on the race environment:
   
   python agent.py
   
   This will:
   - Train a SAC agent on the race environment
   - Save animations every N episodes to "animations_restful/" folder
   - Display training progress and performance metrics
   - Save the trained model


2. CUSTOM ENVIRONMENT (Infinite Valley):
   ---------------------------------------
   To train the agent on the custom infinite valley environment:
   
   cd Custom_Environment
   python agent.py
   
   This will:
   - Train a SAC agent on the infinite valley terrain
   - Generate visualizations of the agent's navigation
   - Save training results to "training_results.txt"


PROJECT STRUCTURE
-----------------
Final/
├── agent.py                    # Main training script for race environment
├── race_acc.py                 # Race environment implementation
├── requirements.txt            # Python dependencies
├── README.txt                  # This file
├── Custom_Environment/
│   ├── agent.py                # Training script for infinite valley
│   ├── infinite_valley_env.py  # Custom environment implementation
│   └── training_results.txt    # Training logs and results
└── Inference/
    ├── race_acc.py             # Race environment for inference
    └── run_inference.py        # Inference script for trained models


KEY FEATURES
------------
- Race Environment:
  * Vehicle dynamics with acceleration limits
  * Gate-based racing track
  * Continuous action space (velocity and rotation)
  
- Infinite Valley Environment:
  * Unbounded 2D terrain with periodic hills/valleys
  * Gravity effects and state-dependent friction
  * Camera follows agent for infinite exploration


OUTPUT FILES
------------
After running the code, you will find:
- animations_restful/: GIF animations of agent performance (race environment)
- Trained model files (*.zip)
- Training logs and metrics
- Custom_Environment/training_results.txt: Performance logs


TROUBLESHOOTING
---------------
1. If you encounter "No module named" errors:
   - Make sure all dependencies are installed: pip install -r requirements.txt

2. If animations are not generated:
   - Ensure ffmpeg is installed on your system
   - Check write permissions in the output directories

3. If training is slow:
   - The code uses PyTorch. For GPU acceleration, install CUDA-enabled PyTorch
   - Reduce the number of training steps or episodes

4. Import errors with race_acc:
   - Make sure you're in the correct directory when running agent.py


NOTES
-----
- Training can take significant time depending on your hardware
- GIF generation happens periodically during training
- You can interrupt training with Ctrl+C and resume later by loading the saved model
- Default hyperparameters are tuned for the respective environments

