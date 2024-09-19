# Standard library to manage files
import os
# Used to display the type of the variables for readability and tests
from typing import Dict, List, Tuple
# Numpy is a common library to make calculus on arrays of numbers
import numpy as np
# Pandas is a common library to manage tabular data (think Excel)
import pandas as pd
# Seaborn and matplotlib are used to plot figures
import matplotlib.pyplot as plt
from matplotlib.image import imread
%matplotlib inline
import seaborn as sns

# PyTorch is a deep learning library
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
!pip install torchinfo
from torchinfo import summary
