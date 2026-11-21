"""Minimal real Gymnasium integration. A random policy is not trained AI."""
import gymnasium as gym
import taklatoon


def main():
    env = gym.make("TaklaToon-v0", level="gap", randomize=True)
    env.action_space.seed(42)
    try:
        obs, info = env.reset(seed=42)
        total_reward = 0.
        while True:
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, info = env.step(action)
            # Future world model: learn from (obs, action, next_obs).
            # Do not read simulator internals or evaluator metadata.
            total_reward += reward
            obs = next_obs
            if terminated or truncated:
                print(info["event"], "return:", round(total_reward, 3))
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()

