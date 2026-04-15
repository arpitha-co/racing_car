# RACING ENVIRONMENT

OVERVIEW
--------
This project contains reinforcement learning agents trained on:
1. Race Environment (race_acc.py) - A racing track with gates


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
1. Install Python dependencies:
   pip install -r requirements.txt

2. Install ffmpeg (for video/animation generation):
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



KEY FEATURES
------------
- Race Environment:
  * Vehicle dynamics with acceleration limits
  * Gate-based racing track
  * Continuous action space (velocity and rotation)
  


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

