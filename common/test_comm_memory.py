import unittest
import torch
import numpy as np
from replay_buffer import CommBatchEpisodeMemory, CommMemory


class TestCommBatchEpisodeMemory(unittest.TestCase):
    def test_store_one_episode(self):
        memory = CommBatchEpisodeMemory(
            continuous_actions=False, n_actions=3, n_agents=2
        )
        obs = {"agent0": np.array([1, 2]), "agent1": np.array([3, 4])}
        state = np.array([5, 6])
        actions = [1, 2]
        reward = 7.0
        memory.store_one_episode(obs, state, actions, reward)

        self.assertEqual(memory.n_step, 1)

    def test_clear_memories(self):
        memory = CommBatchEpisodeMemory(
            continuous_actions=False, n_actions=3, n_agents=2
        )
        obs = {"agent0": np.array([1, 2]), "agent1": np.array([3, 4])}
        state = np.array([5, 6])
        actions = [1, 2]
        reward = 7.0
        memory.store_one_episode(obs, state, actions, reward)
        memory.clear_memories()

        self.assertEqual(memory.n_step, 0)

    def test_get_batch_data(self):
        # Test the get_batch_data method
        pass


class TestCommMemory(unittest.TestCase):
    def test_store_episode(self):
        comm_memory = CommMemory()
        batch_memory = CommBatchEpisodeMemory(
            continuous_actions=False, n_actions=3, n_agents=2
        )

        obs = {"agent0": np.array([1, 2]), "agent1": np.array([3, 4])}
        state = np.array([5, 6])
        actions = [1, 2]
        reward = 7.0
        obs_next = {"agent0": np.array([2, 3]), "agent1": np.array([4, 5])}
        state_next = np.array([6, 7])  # Add this line
        batch_memory.store_one_episode(
            obs,
            state,
            actions,
            reward,
            one_obs_next=obs_next,
            one_state_next=state_next,
        )  # Update this line

        comm_memory.store_episode(batch_memory)

        self.assertEqual(comm_memory.get_memory_real_size(), 1)

    def test_sample(self):
        # Test the sample method
        pass

    def test_get_memory_real_size(self):
        # Test the get_memory_real_size method
        pass


if __name__ == "__main__":
    unittest.main()
