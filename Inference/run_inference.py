#!/usr/bin/env python3
"""
Inference runner for saved SAC models.
Runs multiple episodes and can save per-episode MP4 videos.

Example:
  python Inference/run_inference.py --model ../sac_parcour_1_20250101_120000.zip --instance 1 --episodes 3 --save-video --out-dir ./videos
"""
import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

# Allow importing project modules (race_acc.py)
sys.path.append(str(Path(__file__).resolve().parents[1]))

from race_acc import Parcour
from agent_restful import RaceEnvGymAdapter, ScaledActionWrapper
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import SAC
import torch


def make_env(instance_id: int, max_episode_steps: int):
    raw = Parcour()
    env = RaceEnvGymAdapter(raw, instance_id=instance_id)
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    env = ScaledActionWrapper(env)
    return env


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to saved SAC model (.zip)")
    p.add_argument("--instance", type=int, default=1, help="Instance id (used for output filenames)")
    p.add_argument("--max-steps", type=int, default=300, help="Max episode steps")
    p.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"), help="Device for model")
    p.add_argument("--deterministic", action="store_true", help="Use deterministic actions")
    p.add_argument("--render", action="store_true", help="Call env.render() each step")
    p.add_argument("--episodes", type=int, default=1, help="Number of episodes to run")
    p.add_argument("--save-video", action="store_true", help="Save MP4 per episode")
    p.add_argument("--out-dir", default="inference_videos", help="Directory to save videos")
    p.add_argument("--fps", type=int, default=30, help="Frames per second for saved video")
    args = p.parse_args()

    env = make_env(args.instance, args.max_steps)

    print(f"Loading model {args.model} on device {args.device}...")
    model = SAC.load(args.model, env=env, device=args.device)

    out_dir = Path(args.out_dir)
    if args.save_video:
        out_dir.mkdir(parents=True, exist_ok=True)

    # Run multiple episodes
    for ep in range(1, args.episodes + 1):
        obs, info = env.reset()
        step = 0

        # Setup video writer if requested
        writer = None
        fig = None
        if args.save_video:
            fig = plt.figure(1)
            try:
                writer = FFMpegWriter(fps=args.fps, codec="libx264")
            except Exception as e:
                print(f"Video writer init failed ({e}); continuing without saving video")
                writer = None

        print(f"Starting episode {ep}/{args.episodes}...")

        try:
            if writer is not None:
                out_path = out_dir / f"inference_inst{args.instance}_ep{ep:03d}.mp4"
                try:
                    with writer.saving(fig, str(out_path), dpi=100):
                        while True:
                            action, _ = model.predict(obs, deterministic=args.deterministic)
                            obs, reward, terminated, truncated, info = env.step(action)
                            step += 1
                            env.render()
                            fig.canvas.draw()
                            writer.grab_frame()
                            if terminated or truncated:
                                break
                except Exception as e:
                    print(f"Error during video creation: {e}")
            else:
                while True:
                    action, _ = model.predict(obs, deterministic=args.deterministic)
                    obs, reward, terminated, truncated, info = env.step(action)
                    step += 1
                    if args.render:
                        env.render()
                    if terminated or truncated:
                        break

            print(f"Episode {ep} finished after {step} steps")

            # Print per-episode stats from adapter if available
            try:
                gym_env = env
                if hasattr(env, 'env'):
                    inner = env.env
                    if hasattr(inner, 'env'):
                        inner2 = inner.env
                        if hasattr(inner2, 'total_calc_calls'):
                            gym_env = inner2
                print(f"  Total _calc_next_state calls: {getattr(gym_env, 'total_calc_calls', 'N/A')}")
                print(f"  First solve timestep: {getattr(gym_env, 'first_solve_timestep', 'N/A')}")
                print(f"  Best (min) episode steps to solve: {getattr(gym_env, 'best_timestep', 'N/A')}")
                print(f"  Calc calls at best (since first solve): {getattr(gym_env, 'calc_calls_at_best', 'N/A')}")
            except Exception:
                pass

        finally:
            try:
                plt.close(fig)
            except Exception:
                pass

    env.close()


if __name__ == "__main__":
    main()
 


#python Inference/run_inference.py \  --model /home/d33f/drl/sac_parcour_1_20260208_154919.zip \  --instance 1 --episodes 1 --save-video --out-dir ./videos --fps 30 
