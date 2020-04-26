import numpy as np
import random
from client.transit import Transition


class ReplayBuffer(object):
    beta_increment_per_sampling = 0.0001

    def __init__(self, size):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self.epsilon = 0.01
        self.alpha = 0.7
        self.beta = 0.4

        self.max_priority = 1
        self._storage = []
        self._maxsize = int(size)
        self._next_idx = 0
        self.priorities = np.zeros(self._maxsize)

    def __len__(self):
        return len(self._storage)

    def clear(self):
        self._storage = []
        self._next_idx = 0

    def add(self, transit: Transition, priority):
        if self._next_idx >= len(self._storage):
            self._storage.append(transit)
        else:
            self._storage[self._next_idx] = transit

        pr = self._get_priority(priority)
        self.priorities[self._next_idx] = pr
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def make_index(self, batch_size):
        return [
            random.randint(0, len(self._storage) - 1)
            for _ in range(batch_size)
        ]

    def make_latest_index(self, batch_size):
        idx = [
            (self._next_idx - 1 - i) % self._maxsize for i in range(batch_size)
        ]
        np.random.shuffle(idx)
        return idx

    def sample_index(self, idxes):
        transits = []
        for i in idxes:
            transits.append(self._storage[i])
        return transits

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        if batch_size > 0:
            idxes = self.make_index(batch_size)
        else:
            idxes = range(0, len(self._storage))
        return self._encode_sample(idxes)

    def collect(self):
        return self.sample(-1)

    def sampling_data_prioritized(self, batch_size):

        if self.__len__() < 3200:
            return [], []
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
        sampling_probabilities = (
            self.priorities[
                : self.__len__()
            ]/self.priorities[
                : self.__len__()
            ].sum()
        )
        batch_data_idxs = np.random.choice(
            list(range(self.__len__())),
            size=batch_size,
            p=sampling_probabilities
        )

        sampling_probabilities = sampling_probabilities[batch_data_idxs]
        intermediate_importance_weight = (
            self.__len__() * sampling_probabilities
                                         ) ** -self.beta

        max_of_weights = intermediate_importance_weight.max()
        importance_sampling = intermediate_importance_weight / max_of_weights
        return batch_data_idxs, importance_sampling

    def _get_priority(self, delta):
        return (delta + self.epsilon) ** self.alpha

    def update_priority(self, batch_data_idxs, new_priorities):
        new_priorities = self._get_priority(new_priorities)
        for i in range(len(batch_data_idxs)):
            self.priorities[batch_data_idxs[i]] = new_priorities[i]

