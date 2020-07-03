import random
from typing import Any, Tuple, List
import numpy as np
import torch


# https://raw.githubusercontent.com/rlcode/per/master/SumTree.py
class SumTree:
    """ SumTree: a binary tree data structure where the parent’s value is the sum of its children """
    write_pos = 0

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    # update to the root node
    def _propagate(self, idx: int, change: float):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, index: int, s: float) -> int:
        left = 2 * index + 1
        right = left + 1

        if left >= len(self.tree):
            return index

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self) -> float:
        return self.tree[0]

    # store priority and sample
    def add(self, priority: float, data: Any):
        index = self.write_pos + self.capacity - 1

        self.data[self.write_pos] = data
        self.update(index, priority)

        self.write_pos += 1
        if self.write_pos >= self.capacity:
            self.write_pos = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, index: int, priority: float):
        change = priority - self.tree[index]

        self.tree[index] = priority
        self._propagate(index, change)

    # get priority and sample
    def get(self, s: float) -> Tuple[int, float, Any]:
        index = self._retrieve(0, s)
        data_index = index - self.capacity + 1

        return index, self.tree[index], self.data[data_index]


# https://raw.githubusercontent.com/rlcode/per/master/prioritized_memory.py
class PrioritizedReplayBuffer:
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, max_size: int, num_transition_dims: int):
        self.num_transition_dims = num_transition_dims
        self.tree = SumTree(max_size)
        self.capacity = max_size
        self.last_indices: List[int] = []

    def _get_priority(self, error: float) -> float:
        return (np.abs(error) + self.e) ** self.a

    def size(self) -> int:
        return self.tree.n_entries

    def add(self, error: float, transition: tuple):
        assert len(transition) == self.num_transition_dims
        priority = self._get_priority(error)
        self.tree.add(priority, transition)

    def add_many(self, errors: List[float], transitions: List[tuple]):
        for error, transition in zip(errors, transitions):
            self.add(error, transition)

    def get_batch(self, batch_size: int) -> List[torch.Tensor]:
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        cols = [[] for _ in range(self.num_transition_dims)]
        segment = self.tree.total() / batch_size
        self.last_indices = []
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (index, priority, transition) = self.tree.get(s)

            self.last_indices.append(index)

            for j, val in enumerate(transition):
                cols[j].append(val)

        results = []
        for col in cols:
            # There are always values, except one case: the goal for the top level are None, and therefore can't be changed to into
            # a tensor of floats. In that case, they stay as a list of None's, and they will be ignore by the critic and actor
            if col[0] is not None:
                values_tensor = torch.as_tensor(col, dtype=torch.float32)
            else:
                values_tensor = col
            results.append(values_tensor)

        return results

    def update(self, index: int, error: float):
        priority = self._get_priority(error)
        self.tree.update(index, priority)
