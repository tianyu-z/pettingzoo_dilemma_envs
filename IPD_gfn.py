from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_checker import check_env
from dilemma_pettingzoo import env, parallel_env
from pettingzoo.test import parallel_api_test


if __name__ == "__main__":
    # Testing the parallel algorithm alone
    env_parallel = parallel_env()
    parallel_api_test(env_parallel)  # This works!

    # Testing the environment with the wrapper
    env = env()

    # ERROR: AssertionError: The observation returned by the `reset()` method does not match the given observation space
    check_env(env)

    # Model initialization
    model = PPO("MlpPolicy", env, verbose=1)

    # ERROR: ValueError: could not broadcast input array from shape (20,20) into shape (20,)
    model.learn(total_timesteps=10_000)
