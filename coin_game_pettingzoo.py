# reference
# https://github.com/arjun-prakash/pz_dilemma
# https://github.com/Farama-Foundation/PettingZoo/blob/master/pettingzoo/classic/rps/rps.py

import gymnasium
import numpy as np
import random
from gymnasium.spaces import Discrete
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn
SEED = 0

def env(**kwargs):
    env = raw_env(**kwargs)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


parallel_env = parallel_wrapper_fn(env)


class raw_env(AECEnv):
    """Two-player environment for rock paper scissors.
    Expandable environment to rock paper scissors lizard spock action_6 action_7 ...
    The observation is simply the last opponent action.
    """

    metadata = {
        "render_modes": ["human"],
        "name": "coin_game_v0",
        "is_parallelizable": True,
    }

    def __init__(
        self,
        num_actions=4,
        max_cycles=15,
        render_mode=None,
        nb_players=2,
        grid_size=3,
        randomize_coin=False,
    ):
        self.nb_players = nb_players
        self.max_cycles = max_cycles
        self.render_mode = "human"
        self.name = "coin_game_v0"
        self.randomize_coin = randomize_coin
        self._moves = [
            np.array([0, 1]),  # right
            np.array([0, -1]),  # left
            np.array([1, 0]),  # down
            np.array([-1, 0]),  # up
        ]
        # none is last possible action, to satisfy discrete action space
        self._none = np.array([0, 0])

        self.agents = ["player_" + str(r) for r in range(self.nb_players)]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(zip(self.agents, list(range(self.nb_players))))
        self.action_spaces = {agent: Discrete(num_actions) for agent in self.agents}
        self.observation_spaces = {
            agent: Discrete(1 + num_actions) for agent in self.agents
        }
        self.grid_size = grid_size
        self.ob_space_shape = [
            self.nb_players * 2,
            self.grid_size,
            self.grid_size,
        ]  # *2 because we need the pos of the coin and the player itself, they are paired
        self.grids = [
            [i, j] for i in range(self.grid_size) for j in range(self.grid_size)
        ]

        self.state = {
            agent: {"px": -1, "py": -1, "cx": -1, "cy": -1, "hc": 0}
            for agent in self.agents
        }  # init the location of agents and the coin player_x, player_y, coin_x, coin_y, has_coin
        self.observations = {agent: None for agent in self.agents}
        self.coin_pos = -1 * np.ones(2, dtype=np.int8)
        self.coin_pos_old = self.coin_pos.copy()
        self.player_pos = -1 * np.ones(
            (self.nb_players, 2)
        )  # the x-coord and y-coord of the player
        self.player_coin = np.random.randint(
            self.nb_players
        )  # the generated coin's color
        self.state[self.agents[self.player_coin]][
            "hc"
        ] = 1  # the player who has the coin
        self.render_mode = render_mode

        # aux variables during steps
        self.actions_taken = {agent: None for agent in self.agents}
        self.generate_new_coin = False

        self.reinit()

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def _generate_coin(self, randomize=False):
        self.player_coin_old = self.player_coin

        if randomize:
            self.player_coin = np.random.randint(
                self.nb_players
            )  # next coin belong to a random agent
        else:
            self.player_coin = (
                1 + self.player_coin
            ) % self.nb_players  # next coin belong to next agent
        self.grids_copy = self.grids.copy()
        for j in range(self.nb_players):
            self.grids_copy.remove(list(self.player_pos[j, :]))
        random.shuffle(self.grids_copy)
        self.coin_pos = np.array(self.grids_copy[0])
        return

    def _generate_state(self):
        for j in range(self.nb_players):
            self.state[self.agents[j]]["px"] = self.player_pos[j][0]
            self.state[self.agents[j]]["py"] = self.player_pos[j][1]
            if self.player_coin == j:
                self.state[self.agents[j]]["cx"] = self.coin_pos[0]
                self.state[self.agents[j]]["cy"] = self.coin_pos[1]
                self.state[self.agents[j]]["hc"] = 1
        return

    def _generate_observation(self):
        state_and_action_and_reward = self.state.copy()
        for a in self.agents:
            state_and_action_and_reward[a]["action"] = self.actions_taken[a]
            state_and_action_and_reward[a]["reward"] = self.rewards[a]
        tmp = []
        for a in self.agents:
            tmp += list(state_and_action_and_reward[a].values())
        for a in self.agents:
            self.observations[a] = tmp.copy()
        del state_and_action_and_reward
        return

    def reinit(self):
        self.agents = self.possible_agents[:]
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        self.player_pos = -1 * np.ones((self.nb_players, 2))
        self.coin_pos = -1 * np.ones(2, dtype=np.int8)
        # self.state = {agent: self._none for agent in self.agents}

        # generate player_pos
        self.grids_copy = self.grids.copy()
        random.shuffle(self.grids_copy)
        self.player_pos[:, :] = np.array(self.grids_copy[: self.nb_players])
        # generate coin pos
        self._generate_coin(self.randomize_coin)

        # aux variables during steps
        self.generate_new_coin = False
        self.actions_taken = {agent: None for agent in self.agents}

        self._generate_state()
        self.observations = {agent: None for agent in self.agents}
        self.num_moves = 0

    def render(self):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return
        print("This is a {} round".format(self.num_moves))
        print("Coin (before taken) position: {}".format(self.coin_pos))
        print("Coin (before taken) belongs to: {}".format(self.player_coin))
        agent = self.agent_selection
        if len(self.agents) == self.nb_players:
            # print("Players information: ")
            print(
                "Agent {} position before action: {} ".format(
                    agent, self.player_pos_old[self.agent_name_mapping[agent], :]
                )
            )
            print(
                "Agent {} action: {} ".format(
                    agent, self._moves[self.actions_taken[agent]]
                )
            )
            print(
                "Agent {} position after action: {} ".format(
                    agent, self.player_pos[self.agent_name_mapping[agent], :]
                )
            )
            print("Agent {} reward after action: {} ".format(agent, self.rewards[agent]))
        else:
            print("Game over")
        print("\n")

    def observe(self, agent):
        # observation of one agent is the previous state of the other
        return np.array(self.observations[agent])

    def close(self):
        pass

    def reset(self, seed=SEED, return_info=False, options=None):
        self.reinit()

    def _same_pos(self, x, y):
        return (x == y).all()

    def step(self, action):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return

        agent = self.agent_selection
        
        # self.state[self.agent_selection] = action
        if self._agent_selector.is_first():
            self.rewards = {agent: 0 for agent in self.agents}
        self.actions_taken[agent] = action
        self.player_pos_old = self.player_pos.copy()
        potential_position = (
            self.player_pos[self.agent_name_mapping[agent], :] + self._moves[action]
        ) % self.grid_size
        if potential_position.tolist() not in self.player_pos.tolist():
            self.player_pos[self.agent_name_mapping[agent], :] = potential_position
        # if this grid already has an agent, stay still

        # compute rewards
        if self.player_coin == self.agent_name_mapping[agent]:
            if self._same_pos(
                self.player_pos[self.agent_name_mapping[agent]], self.coin_pos
            ):
                self.generate_new_coin = True
                self.rewards[agent] += 1
            for k in range(self.nb_players):
                if k != self.agent_name_mapping[agent]:
                    if self._same_pos(self.player_pos[k], self.coin_pos):
                        self.generate_new_coin = True
                        self.rewards[agent] -= 2
                        self.rewards["player_" + str(k)] += 1

        if self.render_mode == "human":
            self.render()

        # when all agent is done, update state and generate observations
        if self._agent_selector.is_last():
            # if a coin is collected and all agents finish their actions, regenerate the coin
            self.coin_pos_old = self.coin_pos.copy()
            if self.generate_new_coin:
                self._generate_coin((self.randomize_coin))
            self.num_moves += 1
            self.truncations = {
                agent: self.num_moves >= self.max_cycles for agent in self.agents
            }
            self._generate_state()
            # observe the current states
            self._generate_observation()  # each agent knows the all the state, action and reward of all the agents

        self._cumulative_rewards[self.agent_selection] = 0
        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()


if __name__ == "__main__":
    if SEED is not None:
        random.seed(SEED)
        np.random.seed(SEED)
    # from pettingzoo.test import parallel_api_test

    env = parallel_env(render_mode="human")
    # parallel_api_test(env, num_cycles=1000)

    # Reset the environment and get the initial observation
    obs = env.reset()
    nb_agent = 2
    # Run the environment for 10 steps
    for _ in range(10):
        # Sample a random action
        actions = {"player_"+str(i): np.random.randint(4) for i in range(nb_agent)}

    #     # Step the environment and get the reward, observation, and done flag
        observations, rewards, terminations, truncations, infos = env.step(actions)

    #     # Print the reward
        # print(rewards)

    #     # If the game is over, reset the environment
        if terminations["player_0"]:
            obs = env.reset()
    