import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

import os
import torch

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter



from race_acc import *


class SaveGifEveryNEpisodesCallback(BaseCallback):
    def __init__(
        self,
        make_env_fn,
        save_dir: str = "animations_restful",
        every_episodes: int = 1000,
        rollout_max_steps: int = 500,
        fps: int = 30,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.make_env_fn = make_env_fn
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.every_episodes = int(every_episodes)
        self.rollout_max_steps = int(rollout_max_steps)
        self.fps = int(fps)
        self.episode_count = 0
       

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if isinstance(infos, (list, tuple)) and len(infos) > 0:
            info0 = infos[0]
            if isinstance(info0, dict) and "episode" in info0:
                self.episode_count += 1
                if self.every_episodes > 0 and (self.episode_count % self.every_episodes) == 0:
                    self._save_gif(self.episode_count)
        return True

    def _save_gif(self, episode_idx: int) -> None:
        """Kept name for compatibility; saves MP4 (more reliable than GIF)."""

        self.save_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.save_dir / f"parcour_ep_{episode_idx:07d}.mp4"

        eval_env = self.make_env_fn()
        try:
            obs, _ = eval_env.reset()

            # race_acc.py draws into figure(1)
            fig = plt.figure(1)

            try:
                writer = FFMpegWriter(fps=self.fps, codec="libx264")
            except Exception as e:
                if self.verbose:
                    print(
                        "[SaveGifEveryNEpisodesCallback] MP4 saving requires ffmpeg. "
                        f"Could not initialize writer ({e}); skipping MP4 save"
                    )
                return

            try:
                with writer.saving(fig, str(out_path), dpi=100):
                    for _ in range(self.rollout_max_steps):
                        action, _ = self.model.predict(obs, deterministic=True)
                        obs, reward, terminated, truncated, info = eval_env.step(action)
                        eval_env.render()
                        fig.canvas.draw()
                        writer.grab_frame()

                        if bool(terminated) or bool(truncated):
                            break

                if self.verbose:
                    print(f"[SaveGifEveryNEpisodesCallback] Saved {out_path}")
            except Exception as e:
                if self.verbose:
                    print(f"[SaveGifEveryNEpisodesCallback] Failed to save MP4 ({e})")
        finally:
            try:
                eval_env.close()
            except Exception:
                pass


class TensorboardEpisodeStatsCallback(BaseCallback):
    """Logs raw per-episode stats (reward/length) to TensorBoard."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.episode_count = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if not isinstance(infos, (list, tuple)):
            return True

        for info in infos:
            if not isinstance(info, dict):
                continue
            ep_info = info.get("episode")
            if not isinstance(ep_info, dict):
                continue

            # Monitor wrapper populates: {"r": ep_reward, "l": ep_len, "t": ep_time}
            self.episode_count += 1
            if "r" in ep_info:
                self.logger.record("episode/raw_reward", float(ep_info["r"]), exclude=("stdout",))
            if "l" in ep_info:
                self.logger.record("episode/length", float(ep_info["l"]), exclude=("stdout",))
            self.logger.record("episode/index", float(self.episode_count), exclude=("stdout",))

            # Flush immediately so you see it in TensorBoard even for short runs.
            self.logger.dump(step=self.num_timesteps)

        return True


class RaceEnvGymAdapter(gym.Env):
    """
    Adapts the professor's RaceEnv/Parcour to Gymnasium API
    WITHOUT modifying the original environment.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        race_env,
        max_pos: float = 50.0,
        progress_reward_scale: float = 0.1,
        lookahead_gates: int = 4,
        instance_id: int = 1,
    ):
        super().__init__()
        self.race_env = race_env
        self.max_pos = float(max_pos)
        self.progress_reward_scale = float(progress_reward_scale)
        self.lookahead_gates = int(max(0, lookahead_gates))
        self.instance_id = instance_id


        # Observation:
        # base: [x, y, theta, v_trans, v_rot]
        # plus: lookahead gate centers in robot-frame: [dx1, dy1, dx2, dy2, ...]
        gate_feat_limit = 2.0 * self.max_pos
        extra = 2 * self.lookahead_gates
        low = np.array(
            [-self.max_pos, -self.max_pos, -np.pi, 0.0, -VEL_ROT_LIMIT] + ([-gate_feat_limit] * extra),
            dtype=np.float32,
        )
        high = np.array(
            [self.max_pos, self.max_pos, np.pi, VEL_TRANS_LIMIT, VEL_ROT_LIMIT] + ([gate_feat_limit] * extra),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Physical action space (accelerations expected by race_acc.py)
        self.action_space = spaces.Box(
            low=np.array([-ACC_TRANS_LIMIT, -ACC_ROT_LIMIT], dtype=np.float32),
            high=np.array([ACC_TRANS_LIMIT, ACC_ROT_LIMIT], dtype=np.float32),
            dtype=np.float32,
        )

        self._prev_dist_to_gate = None
        self._last_gate_idx = None

        self.total_calc_calls = 0
        self.step_after_solved = 0
        self.best_timestep = 0
        self.calc_calls_at_best = 0  # _calc_next_state calls when best_timestep was achieved
        self.first_solve_timestep = 0  # When first solved
        self.episode_steps = 0  # Steps in current episode
        self.episode_number = 0  # Current episode number
        self.episode_log_file = f"episode_steps_{self.instance_id}.csv"
        
        # Initialize CSV file with header (fresh per instance)
        if not os.path.exists(self.episode_log_file):
            with open(self.episode_log_file, "w") as f:
                f.write("episode,steps\n")
        self.first_solved = False

    def _gate_center_world(self, gate_idx: int):
        gates = self.race_env.get_gates()
        if gate_idx < 0 or gate_idx >= len(gates):
            return None
        p = np.asarray(gates[gate_idx][0], dtype=np.float32)
        q = np.asarray(gates[gate_idx][1], dtype=np.float32)
        return 0.5 * (p + q)

    def _future_gate_centers_world(self, start_gate_idx: int):
        # Returns up to lookahead_gates centers starting at start_gate_idx.
        centers = []
        for k in range(self.lookahead_gates):
            c = self._gate_center_world(start_gate_idx + k)
            if c is None:
                break
            centers.append(c)
        return centers

    def _sanitize_obs(self, obs: np.ndarray) -> np.ndarray:
        base = np.asarray(obs, dtype=np.float32).reshape(-1)
        # Replace NaN/Inf to avoid policy NaNs
        base = np.nan_to_num(base, nan=0.0, posinf=self.max_pos, neginf=-self.max_pos)

        # Wrap heading to [-pi, pi]
        base[2] = (base[2] + np.pi) % (2.0 * np.pi) - np.pi

        if self.lookahead_gates <= 0:
            out = base
        else:
            x, y, theta = float(base[0]), float(base[1]), float(base[2])
            gi = int(self.race_env.get_gate_idx())
            centers = self._future_gate_centers_world(gi)

            c = float(np.cos(theta))
            s = float(np.sin(theta))
            rel = []
            for center in centers:
                dx_w = float(center[0] - x)
                dy_w = float(center[1] - y)
                # world -> robot frame (rotate by -theta)
                dx_r = c * dx_w + s * dy_w
                dy_r = -s * dx_w + c * dy_w
                rel.extend([dx_r, dy_r])

            # Pad if fewer gates remain
            missing = self.lookahead_gates - len(centers)
            if missing > 0:
                rel.extend([0.0] * (2 * missing))

            out = np.concatenate([base, np.asarray(rel, dtype=np.float32)], axis=0)

        out = np.clip(out, self.observation_space.low, self.observation_space.high)
        return out.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Log previous episode data (except for the very first episode)
        if self.episode_number > 0:
            with open(self.episode_log_file, "a") as f:
                f.write(f"{self.episode_number},{self.episode_steps}\n")
                f.flush()  # Force write to disk
        
        # Start new episode
        self.episode_number += 1
        self.episode_steps = 0  # Reset episode counter
        
        obs = self.race_env.reset()
        self._last_gate_idx = int(self.race_env.get_gate_idx())
        return self._sanitize_obs(obs), {}

    def step(self, action):
        self.total_calc_calls += 1
        self.episode_steps += 1  # Track steps in this episode
        prev_gate_idx = int(self.race_env.get_gate_idx())

        action = np.asarray(action, dtype=np.float32).reshape(2,)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        obs, base_reward, done = self.race_env.step(action)

        new_gate_idx = int(self.race_env.get_gate_idx())
        self._last_gate_idx = new_gate_idx

        terminated = bool(done)
        truncated = False

        # --------------------------------------------------
        # Gate information
        # --------------------------------------------------
        try:
            gates = self.race_env.get_gates()
            num_gates = len(gates)
        except Exception:
            gates = []
            num_gates = 0

        gate_bonus = 0.0
        shaped = 0.0

        # --------------------------------------------------
        # 1) Increasing reward for passing gates
        # --------------------------------------------------
        if num_gates > 0 and new_gate_idx > prev_gate_idx:
            for passed_gate in range(prev_gate_idx, new_gate_idx):
                # Gate bonus mid from RESTFUL-SWEEP-9
                gate_bonus += 39.031593837280425

                # Final gate bonus from RESTFUL-SWEEP-9
                if passed_gate == num_gates - 1:
                    gate_bonus += 191.86987543506496

        # --------------------------------------------------
        # 2) Dense reward for moving toward NEXT gate center
        # --------------------------------------------------
        if num_gates > 0 and new_gate_idx < num_gates:
            center = self._gate_center_world(new_gate_idx)

            if center is not None:
                car_x, car_y = float(obs[0]), float(obs[1])
                dist = np.linalg.norm(center - np.array([car_x, car_y]))

                if self._prev_dist_to_gate is not None:
                    progress = self._prev_dist_to_gate - dist
                    shaped += 2.0 * progress   # <-- main dense signal

                self._prev_dist_to_gate = dist
        else:
            self._prev_dist_to_gate = None

        # --------------------------------------------------
        # 3) Step penalty from RESTFUL-SWEEP-9
        # --------------------------------------------------
        shaped -= 0.022383304303927303

        # --------------------------------------------------
        # 4) Boundary penalty from RESTFUL-SWEEP-9
        # --------------------------------------------------
        car_x, car_y = float(obs[0]), float(obs[1])
        if abs(car_x) > 45.0 or abs(car_y) > 45.0:  # Near boundary
            shaped -= 0.03503473298780898

        info = {
            "gate_idx": new_gate_idx,
            "gate_bonus": float(gate_bonus),
            "shaped_reward": float(shaped),
        }

        total_reward = base_reward + gate_bonus + shaped
        if num_gates > 0 and new_gate_idx >= num_gates:
            if not self.first_solved:
                self.first_solved = True
                self.first_solve_timestep = self.total_calc_calls

            # Track best solve time (only when actually solved this step)
            if self.best_timestep == 0 or self.episode_steps < self.best_timestep:
                self.best_timestep = self.episode_steps
                # Calls since first solve (not from start)
                self.calc_calls_at_best = self.total_calc_calls - self.first_solve_timestep

        if self.first_solved:
            self.step_after_solved += 1

        return (
            self._sanitize_obs(obs),
            float(total_reward),
            terminated,
            truncated,
            info,
        )


    def render(self):
        self.race_env.plot()


class ScaledActionWrapper(gym.ActionWrapper):
    """
    Scales SAC actions from [-1, 1] to physical accelerations
    """

    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([ 1.0,  1.0], dtype=np.float32),
        )

    def action(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(2,)
        acc_trans = float(action[0]) * ACC_TRANS_LIMIT
        acc_rot   = float(action[1]) * ACC_ROT_LIMIT
        return np.array([acc_trans, acc_rot], dtype=np.float32)
    
class SoftActorCriticAgent:
    def __init__(self, instance_id: int = 1):
        self.instance_id = instance_id
        self.env = self._make_env()

    def _make_env(self):
        raw_env = Parcour()
        gym_env = RaceEnvGymAdapter(raw_env, instance_id=self.instance_id)
        gym_env = TimeLimit(gym_env, max_episode_steps=300)
        env = ScaledActionWrapper(gym_env)
        return Monitor(env)
    
    def train(self):
        requested_device = os.environ.get("SB3_DEVICE", "auto").strip().lower()
        if requested_device in {"auto", ""}:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = requested_device

        print(f"[SAC] Using device: {device}")

        callback = SaveGifEveryNEpisodesCallback(
            make_env_fn=self._make_env,
            save_dir=f"animations_restful_{self.instance_id}",
            every_episodes=10,
            rollout_max_steps=500,
            fps=30,
            verbose=1,
        )

        tb_episode_callback = TensorboardEpisodeStatsCallback(verbose=0)

        # RESTFUL-SWEEP-9 parameters - WORLD RECORD 85 steps
        net_arch_map = {
            "small": [64, 64],
            "medium": [128, 128], 
            "large": [256, 256],
            "xlarge": [512, 512]
        }
        
        model = SAC(
            policy="MlpPolicy",
            env=self.env,
            learning_rate=0.00027686633662801606,
            buffer_size=1000000,
            batch_size=64,
            gamma=0.9316816426199078,
            tau=0.028386620754404093,
            ent_coef="auto_0.1",
            train_freq=1,
            gradient_steps=4,
            learning_starts=4822,
            policy_kwargs=dict(net_arch=net_arch_map["large"]),
            verbose=1,
            tensorboard_log="./tensorboard_logs/",
            device=device,
        )

        model.learn(
            total_timesteps=150_000,
            tb_log_name=f"SAC_Parcour_{self.instance_id}",
            log_interval=10,
            callback=[callback, tb_episode_callback],
        )   
        
        # Save model with timestamp
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"sac_parcour_{self.instance_id}_{timestamp}"
        model.save(model_filename)
        print(f"Model saved as: {model_filename}.zip")
        
        return model
    




# =========================================================
# Entry point
# =========================================================

if __name__ == "__main__":
    NUM_INSTANCES = 10

    # Per-instance results
    all_first_solve_calls = []   # metric 1: _calc_next_state calls at first solve
    all_best_episode_steps = []  # metric 2: minimum episode steps to solve
    all_calls_at_best = []       # metric 3: _calc_next_state calls when best was achieved

    for i in range(1, NUM_INSTANCES + 1):
        print(f"\n{'='*60}")
        print(f"  Instance {i}/{NUM_INSTANCES}")
        print(f"{'='*60}")

        agent = SoftActorCriticAgent(instance_id=i)
        model = agent.train()

        # Unwrap: Monitor -> ScaledActionWrapper -> TimeLimit -> RaceEnvGymAdapter
        gym_env = agent.env.env.env.env

        solved = gym_env.first_solved
        first_solve = gym_env.first_solve_timestep
        best_steps  = gym_env.best_timestep
        calls_best  = gym_env.calc_calls_at_best

        print(f"\n--- Instance {i} results ---")
        print(f"  Solved: {solved}")
        print(f"  1) _calc_next_state calls at first solve : {first_solve}")
        print(f"  2) Min episode steps to solve            : {best_steps}")
        print(f"  3) _calc_next_state calls at best solve  : {calls_best}")

        if solved:
            all_first_solve_calls.append(first_solve)
            all_best_episode_steps.append(best_steps)
            all_calls_at_best.append(calls_best)

    # ------------------------------------------------------------------
    # Averages (only over instances that actually solved)
    # ------------------------------------------------------------------
    n_solved = len(all_first_solve_calls)
    print(f"\n{'='*60}")
    print(f"  SUMMARY  ({n_solved}/{NUM_INSTANCES} instances solved)")
    print(f"{'='*60}")

    if n_solved > 0:
        avg1 = np.mean(all_first_solve_calls)
        avg2 = np.mean(all_best_episode_steps)
        avg3 = np.mean(all_calls_at_best)
        print(f"  1) Avg _calc_next_state calls at first solve : {avg1:.2f}")
        print(f"  2) Avg min episode steps to solve            : {avg2:.2f}")
        print(f"  3) Avg _calc_next_state calls at best solve  : {avg3:.2f}")
    else:
        avg1 = avg2 = avg3 = float('nan')
        print("  No instance solved the environment.")

    # Save to file
    with open("training_results.txt", "w") as f:
        f.write(f"Instances solved: {n_solved}/{NUM_INSTANCES}\n\n")
        f.write(f"Per-instance details (solved only):\n")
        for idx, (fs, bs, cb) in enumerate(
            zip(all_first_solve_calls, all_best_episode_steps, all_calls_at_best), 1
        ):
            f.write(f"  Instance {idx}: first_solve_calls={fs}, best_steps={bs}, calls_at_best={cb}\n")
        f.write(f"\nAverages:\n")
        f.write(f"  1) Avg _calc_next_state calls at first solve : {avg1:.2f}\n")
        f.write(f"  2) Avg min episode steps to solve            : {avg2:.2f}\n")
        f.write(f"  3) Avg _calc_next_state calls at best solve  : {avg3:.2f}\n")

    print("\nResults saved to training_results.txt")
