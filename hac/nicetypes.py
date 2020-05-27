from typing import Tuple
import numpy as np

NumpyArray = np.ndarray
#                          State        Goal       Action
Level1Transition = Tuple[NumpyArray, NumpyArray, NumpyArray]
Transition = Tuple
