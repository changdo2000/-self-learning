import math
import numpy as np

def sigmoid(s):
    return 1 / (1 + np.exp(-s))

def relu(s):
    return max(0, s)