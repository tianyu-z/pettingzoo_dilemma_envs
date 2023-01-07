# reference
# https://github.com/arjun-prakash/pz_dilemma
# https://github.com/Farama-Foundation/PettingZoo/blob/master/pettingzoo/classic/rps/rps.py

import gymnasium
import numpy as np
from gymnasium.spaces import Discrete

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn

from games import Game, Prisoners_Dilemma, Samaritans_Dilemma, Stag_Hunt, Chicken


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
        "name": "simple_pd_v0",
        "is_parallelizable": True,
        "render_fps": 2,
    }

    def __init__(self, game="pd", num_actions=2, max_cycles=15, render_mode=None):
        self.max_cycles = max_cycles
        GAMES = {
            "pd": Prisoners_Dilemma(),
            "sd": Samaritans_Dilemma(),
            "stag": Stag_Hunt(),
            "chicken": Chicken(),
        }
        self.render_mode = "human"
        self.name = "simple_pd_v0"
        self.game = GAMES[game]

        self._moves = self.game.moves
        # none is last possible action, to satisfy discrete action space
        self._none = self.game.NONE

        self.agents = ["player_" + str(r) for r in range(2)]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(zip(self.agents, list(range(self.num_agents))))
        self.action_spaces = {agent: Discrete(num_actions) for agent in self.agents}
        self.observation_spaces = {
            agent: Discrete(1 + num_actions) for agent in self.agents
        }

        self.render_mode = render_mode

        self.reinit()

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reinit(self):
        self.agents = self.possible_agents[:]
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        self.state = {agent: self._none for agent in self.agents}
        self.observations = {agent: self._none for agent in self.agents}

        self.history = [0] * (2 * 5)

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

        if len(self.agents) == 2:
            string = "Current state: Agent1: {} , Agent2: {}".format(
                self._moves[self.state[self.agents[0]]],
                self._moves[self.state[self.agents[1]]],
            )
        else:
            string = "Game over"
        print(string)

    def observe(self, agent):
        # observation of one agent is the previous state of the other
        return np.array(self.observations[agent])

    def close(self):
        pass

    def reset(self, seed=None, return_info=False, options=None):
        self.reinit()

    def step(self, action):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return

        agent = self.agent_selection

        self.state[self.agent_selection] = action

        # collect reward if it is the last agent to act

        if self._agent_selector.is_last():
            (
                self.rewards[self.agents[0]],
                self.rewards[self.agents[1]],
            ) = self.game.payoff[
                (self.state[self.agents[0]], self.state[self.agents[1]])
            ]

            self.num_moves += 1
            self.truncations = {
                agent: self.num_moves >= self.max_cycles for agent in self.agents
            }

            # observe the current state
            for i in self.agents:
                self.observations[i] = self.state[
                    self.agents[1 - self.agent_name_mapping[i]]
                ]
        else:
            self.state[self.agents[1 - self.agent_name_mapping[agent]]] = self._none
            self._clear_rewards()

        self._cumulative_rewards[self.agent_selection] = 0
        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()

        if self.render_mode == "human":
            self.render()


if __name__ == "__main__":
    from pettingzoo.test import parallel_api_test

    env = parallel_env(render_mode="human")
    parallel_api_test(env, num_cycles=1000)

    # Reset the environment and get the initial observation
    obs = env.reset()

    # Run the environment for 10 steps
    # for _ in range(10):
    #     # Sample a random action
    #     action_a = env.action_spaces["player_0"].sample()
    #     action_b = env.action_spaces["player_1"].sample()
    #     actions = (action_a, action_b)

    #     # Step the environment and get the reward, observation, and done flag
    #     obs, reward, done, _ = env.step(actions)

    #     # Print the reward
    #     print(reward)

    #     # If the game is over, reset the environment
    #     if done:
    #         obs = env.reset()
