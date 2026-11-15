#!/usr/bin/env python3
"""
Evaluate a trained SAC model on the Parcour track.

Runs N episodes (with the environment's actuation noise enabled) and reports
the success rate and the number of steps needed to pass all gates.

Example:
  python Inference/evaluate.py --model models/sac_parcour_best.zip --episodes 50
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from gymnasium.wrappers import TimeLimit  # noqa: E402
from stable_baselines3 import SAC  # noqa: E402

from agent import RaceEnvGymAdapter, ScaledActionWrapper  # noqa: E402
from race_acc import Parcour  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to saved SAC model (.zip)")
    p.add_argument("--episodes", type=int, default=50, help="Number of evaluation episodes")
    p.add_argument("--max-steps", type=int, default=500, help="Step limit per episode")
    p.add_argument("--seed", type=int, default=0, help="Seed for the actuation noise")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    np.random.seed(args.seed)
    env = ScaledActionWrapper(
        TimeLimit(RaceEnvGymAdapter(Parcour(), instance_id=0), max_episode_steps=args.max_steps)
    )
    model = SAC.load(args.model, device=args.device)
    n_gates = len(env.unwrapped.race_env.get_gates())

    successes, steps, gates = 0, [], []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done, t, info = False, 0, {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            t += 1
            done = terminated or truncated
        gates.append(info.get("gate_idx", 0))
        if terminated:
            successes += 1
            steps.append(t)

    print(f"Model:        {args.model}")
    print(f"Episodes:     {args.episodes}")
    print(f"Success rate: {successes}/{args.episodes} ({100 * successes / args.episodes:.0f}%)")
    print(f"Gates passed: {np.mean(gates):.1f}/{n_gates} on average")
    if steps:
        print(f"Steps to finish: median {np.median(steps):.0f}, min {min(steps)}, max {max(steps)}")


if __name__ == "__main__":
    main()
