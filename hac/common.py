import dataclasses
from collections import deque
from typing import Union, List, Tuple

import numpy as np
import torch


def get_tensor(x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    if torch.is_tensor(x):
        return x.float()
    else:
        return torch.tensor(x, dtype=torch.float32)


class ReplayBuffer:
    def __init__(self, max_size: int, num_transition_dims: int):
        self.num_transition_dims = num_transition_dims
        self.buffers = [deque(maxlen=max_size) for _ in range(num_transition_dims)]

    def size(self) -> int:
        # All buffers have the same size, so we abitrarily examine the first one
        return len(self.buffers[0])

    def add(self, transition: tuple):
        assert len(transition) == self.num_transition_dims
        for i, element in enumerate(transition):
            self.buffers[i].append(element)

    def add_many(self, transitions: List[tuple]):
        for transition in transitions:
            self.add(transition)

    def get_batch(self, batch_size: int) -> List[torch.Tensor]:
        indices = np.random.randint(low=0, high=len(self.buffers[0]), size=batch_size)
        results = []
        for buffer in self.buffers:
            values_list = [buffer[index] for index in indices]
            # There are always values, except one case: the goal for the top level are None, and therefore can't be changed to into
            # a tensor of floats. In that case, they stay as a list of None's, and they will be ignore by the critic and actor
            if values_list[0] is not None:
                values_tensor = torch.as_tensor(values_list, dtype=torch.float32)
            else:
                values_tensor = values_list
            results.append(values_tensor)

        return results


def get_range_and_center(low: np.array, high: np.array) -> Tuple[np.ndarray, np.ndarray]:
    bounds_range = (high - low) / 2  # The whole range [low, high] has size (high - low) and the range is half of it
    center = (high + low) / 2  # The center of the interval [low, high] is the average of the extremes
    return bounds_range, center


def json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)

    raise TypeError('Not serializable')