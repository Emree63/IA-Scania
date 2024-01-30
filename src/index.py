import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from constants import *

train = pd.read_csv(TRAINING_PATH)
test = pd.read_csv(TEST_PATH)