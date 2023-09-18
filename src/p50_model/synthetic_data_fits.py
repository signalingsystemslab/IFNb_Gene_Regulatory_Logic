# Generate synthetic data based on p50 training data
# Fit p50 model to each set of synthetic data points
# Evaluate the parameter distributions of the fitted models from all synthetic data sets
# Plot the parameter distributions of the fitted models from all synthetic data sets

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import scipy.optimize as opt
import os
import sys
import time
from p50_model import *
from multiprocessing import Pool

def generate_synthetic_data(training_data, num_datasets, seed):
    num_pts = training_data.shape[0]