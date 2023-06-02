from pettingzoo.classic import tictactoe_v3
import numpy as np

if __name__ == "__main__":
    SEED = 0
    if SEED is not None:
        np.random.seed(SEED)
    # from pettingzoo.test import parallel_api_test

    env = tictactoe_v3.parallel_env(render_mode="human")
    # parallel_api_test(env, num_cycles=1000)

    # Reset the environment and get the initial observation
    obs = env.reset()

    # Run the environment for 10 steps
    for _ in range(10):
        # Sample a random action
        actions = {"player_" + str(i): np.random.randint(9) for i in range(1, 3)}

        # Step the environment and get the reward, observation, and done flag
        observations, rewards, terminations, truncations, infos = env.step(actions)

        # Print the reward
        # print(rewards)
        print("observations: ", observations)
        # If the game is over, reset the environment
        if terminations["player_1"]:
            obs = env.reset()
