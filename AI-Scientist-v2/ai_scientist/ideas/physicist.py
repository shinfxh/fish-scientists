import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

data = pd.read_csv("/home/shinfxh/fish-scientists/AI-Scientist-v2/physics_data.csv")

# continue to visualize the data and run model fitting 
# if there are mulitple trajectory_id present, note that each trajectory_id represents a different motion path started with different initial conditions
# deduce as much information as possible from the data before running parameter sweeping. 
# ignore non-linear effects such as damping. Take gravity to be 9.81.