"""
Coin Game environment.
"""
import os

os.system("pip install gym==0.10.5")
import gym
import numpy as np
import random
from gym.spaces import Discrete, Tuple
from gym.spaces import prng


class CoinGameVec:
    """
    Vectorized Coin Game environment.
    Note: slightly deviates from the Gym API.
    """

    NUM_ACTIONS = 4
    MOVES = [
        np.array([0, 1]),  # right
        np.array([0, -1]),  # left
        np.array([1, 0]),  # down
        np.array([-1, 0]),  # up
    ]

    def __init__(self, max_steps, batch_size, grid_size=3, NUM_AGENTS=2):
        self.NUM_AGENTS = NUM_AGENTS
        self.max_steps = max_steps
        self.grid_size = grid_size
        self.batch_size = batch_size
        # The 4 channels stand for 2 players and 2 coin positions
        self.ob_space_shape = [
            self.NUM_AGENTS * 2,
            grid_size,
            grid_size,
        ]  # *2 because we need the pos of the coin and the player itself, they are paired
        self.grids = [[i, j] for i in range(grid_size) for j in range(grid_size)]
        self.step_count = None
        self.player_pos = np.zeros((self.batch_size, self.NUM_AGENTS, 2))
        self.player_coin = np.random.randint(self.NUM_AGENTS, size=self.batch_size)

    def reset(self):
        self.step_count = 0
        # self.red_coin = prng.np_random.randint(self.NUM_AGENTS, size=self.batch_size)
        # Agent and coin positions
        # self.red_pos = prng.np_random.randint(
        #     self.grid_size, size=(self.batch_size, 2)
        # )  # 2, because it's a 2D grid world
        # self.blue_pos = prng.np_random.randint(
        #     self.grid_size, size=(self.batch_size, 2)
        # )
        self.player_pos = np.zeros((self.batch_size, self.NUM_AGENTS, 2))
        self.coin_pos = np.zeros((self.batch_size, 2), dtype=np.int8)
        for i in range(self.batch_size):
            # Make sure coins don't overlap
            # while self._same_pos(self.red_pos[i], self.blue_pos[i]):
            #     self.blue_pos[i] = prng.np_random.randint(self.grid_size, size=2)

            # tmp = np.arange(len(self.grids))
            # for j in range(self.NUM_AGENTS):
            #     self.player_pos[i, j, :] = self.grids[
            #         np.random.choice(tmp, 1, replace=False)[0]
            #     ]
            self.grids_copy = self.grids.copy()
            random.shuffle(self.grids_copy)
            self.player_pos[i, :, :] = np.array(self.grids_copy[: self.NUM_AGENTS])
            self._generate_coin(i)
        return self._generate_state()

    def _generate_coin(self, i, randomize=False):
        # self.red_coin[i] = 1 - self.red_coin[i]
        if randomize:
            self.player_coin[i] = np.random.randint(
                self.NUM_AGENTS
            )  # next coin belong to a random agent
        else:
            self.player_coin[i] = (
                1 + self.player_coin[i]
            ) % self.NUM_AGENTS  # next coin belong to next agent
        # Make sure coin has a different position than the agents
        # success = 0
        # while success < self.NUM_AGENTS:  # number of agent
        #     self.coin_pos[i] = np.random.randint(self.grid_size, size=(2))
        #     for j in range(self.NUM_AGENTS):
        #         success += 1 - self._same_pos(
        #             self.player_pos[i, j, :], self.coin_pos[i]
        #         )
        self.grids_copy = self.grids.copy()
        for j in range(self.NUM_AGENTS):
            self.grids_copy.remove(list(self.player_pos[i, j, :]))
        random.shuffle(self.grids_copy)
        self.coin_pos[i] = np.array(self.grids_copy[0])
        return

    def _same_pos(self, x, y):
        return (x == y).all()

    # def _generate_state(self):
    #     state = np.zeros([self.batch_size] + self.ob_space_shape)
    #     for i in range(self.batch_size):
    #         state[i, 0, self.red_pos[i][0], self.red_pos[i][1]] = 1 # for each batch, the red player position is set to 1
    #         state[i, 1, self.blue_pos[i][0], self.blue_pos[i][1]] = 1 # for each batch, the blue player position is set to 1
    #         if self.red_coin[i]:
    #             state[i, 2, self.coin_pos[i][0], self.coin_pos[i][1]] = 1 # for each batch, the red coin position is set to 1
    #         else:
    #             state[i, 3, self.coin_pos[i][0], self.coin_pos[i][1]] = 1 # for each batch, the blue coin position is set to 1
    #     return state

    def _generate_state(self):
        state = np.zeros([self.batch_size] + self.ob_space_shape)
        for i in range(self.batch_size):
            for j in range(self.NUM_AGENTS):
                state[
                    i, j, int(self.player_pos[i][j][0]), int(self.player_pos[i][j][1])
                ] = 1
                if self.player_coin[i] == j:
                    state[
                        i, j + self.NUM_AGENTS, self.coin_pos[i][0], self.coin_pos[i][1]
                    ] = 1
        return state

    def step(self, actions):
        for j in range(self.batch_size):
            # ac0, ac1 = actions[j]  # actions.shape = (batch_size, num_agents)
            # assert ac0 in {0, 1, 2, 3} and ac1 in {0, 1, 2, 3}
            for i in range(self.NUM_AGENTS):
                assert actions[j][i] in {0, 1, 2, 3}

            # Move players
            for i in range(self.NUM_AGENTS):
                self.player_pos[j, i] = (
                    self.player_pos[j, i] + self.MOVES[actions[j][i]]
                ) % self.grid_size

        # Compute rewards
        # reward_red, reward_blue = [], []
        reward = np.zeros((self.batch_size, self.NUM_AGENTS))
        for i in range(self.batch_size):
            generate = False
            for j in range(self.NUM_AGENTS):
                if self.player_coin[i] == j:
                    if self._same_pos(self.player_pos[i, j], self.coin_pos[i]):
                        generate = True
                        reward[i, j] = 1
                    for k in range(self.NUM_AGENTS):
                        if k != j:
                            if self._same_pos(self.player_pos[i, k], self.coin_pos[i]):
                                generate = True
                                reward[i, j] -= 2
                                reward[i, k] += 1
            # if self.red_coin[i]:
            #     if self._same_pos(self.red_pos[i], self.coin_pos[i]):
            #         generate = True
            #         reward_red.append(1)
            #         reward_blue.append(0)
            #     elif self._same_pos(self.blue_pos[i], self.coin_pos[i]):
            #         generate = True
            #         reward_red.append(-2)
            #         reward_blue.append(1)
            #     else:
            #         reward_red.append(0)
            #         reward_blue.append(0)

            # else:
            #     if self._same_pos(self.red_pos[i], self.coin_pos[i]):
            #         generate = True
            #         reward_red.append(1)
            #         reward_blue.append(-2)
            #     elif self._same_pos(self.blue_pos[i], self.coin_pos[i]):
            #         generate = True
            #         reward_red.append(0)
            #         reward_blue.append(1)
            #     else:
            #         reward_red.append(0)
            #         reward_blue.append(0)

            if generate:
                self._generate_coin(i)

        # reward = [np.array(reward_red), np.array(reward_blue)]
        self.step_count += 1
        done = np.array(
            [(self.step_count == self.max_steps) for _ in range(self.batch_size)]
        )
        state = self._generate_state()
        return state, reward, done


if __name__ == "__main__":
    env = CoinGameVec(10, 2, NUM_AGENTS=3)

    # Reset the environment and get the initial observation
    obs = env.reset()

    # Run the environment for 10 steps
    for _ in range(10):
        # Sample a random action
        actions = [
            [np.random.randint(env.NUM_ACTIONS) for _ in range(env.NUM_AGENTS)]
            for n in range(env.batch_size)
        ]
        # Step the environment and get the reward, observation, and done flag
        obs, reward, done = env.step(actions)

        # Print the reward
        print(reward)

        # If the game is over, reset the environment
        if done.all():
            obs = env.reset()
