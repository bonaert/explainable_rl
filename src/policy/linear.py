import torch.nn as nn
import torch.nn.functional as F


class SimplePolicy(nn.Module):
    """ Simple policy inspired by Torch's example policy for the Reinforce example code.
    See: https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py
    """

    def __init__(self, input_size: int, output_size: int):
        super(SimplePolicy, self).__init__()
        self.affine1 = nn.Linear(input_size, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(128, output_size)

    def forward(self, x):
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)  # Use softmax to return probabilities
