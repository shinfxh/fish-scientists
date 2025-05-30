import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

data = pd.read_csv("/home/shinfxh/fish-scientists/AI-Scientist-v2/physics_data.csv")

# continue to visualize the data and run model fitting 
# note that each trajectory_id represents a different motion path started with different initial conditions