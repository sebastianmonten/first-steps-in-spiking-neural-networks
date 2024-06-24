import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import warnings
warnings.filterwarnings("ignore")


# My imports
from typing import List, Tuple

# Load original dataset
URL = 'datasets/Iris.csv'
df = pd.read_csv(URL) # pandas.core.frame.DataFrame
df = df.iloc[:,1:]

# Print the first few line sof the dataset
print(df.head())

# Make a copy of the dataframe, delete the ‘species’ column, leaving only the quantitative part of the detaset
df_ = df.drop(columns=['Species']).copy()

# Build a data histogram:
df_.plot.hist(alpha = 0.4, figsize = (12, 4))
plt.legend(title = "Dataset cilumns:", bbox_to_anchor = (1.0, 0.6), loc = 'upper left')
plt.title('Iris dataset', fontsize = 20)
plt.xlabel('Input value', fontsize = 15)
plt.show()


def Gaus_neuron(df , n : int, step: float, s: List[float]) -> Tuple[List[int], List[int]]:
    """
    Function that generates 10 Gaussians for each input feature so that:

    - means of each Gaussian are evenly distributed between the extreme values
    of the range, including the boundaries for each feature 
    ( “Sepal Length”, “Sepal Width”, “Petal Length”, and “Petal Width”)

    - the height of each Gaussian is equal to 1 is the maximum excitation
    value of the presynaptic neuron, from which later we will calculate the
    spike generation latency by presynaptic neuron.

    Args:
        df:   The input DataFrame containing the features.
        n:    The number of Gaussians to generate for each feature.
        step: The step size for generating the x-axis values.
        s:    A list of standard deviations for the Gaussian distributions, one for each feature.

    Returns:
        List[int]: A list where each element is a 2D array representing the Gaussian distributions for a feature.
        List[int]: A list where each element is an array of x-axis values corresponding to the Gaussians of a feature.

    """

    neurons_list = list()
    x_axis_list = list()
    t = 0 # counter to keep track of the current feature index

    for col in df.columns:

        vol = df[col].values # numpy.ndarray
        min_ = np.min(vol)
        max_ = np.max(vol)
        x_axis = np.arange(min_, max_, step) # numpy.ndarray {min_, min_ + step, min_ + 2*step, ... , max_ - step}
        x_axis[0] = min_  # hard code first element to min_
        x_axis[-1] = max_ # hard code last element to max_
        x_axis_list.append(np.round(x_axis, 10)) # x_axis_list gets a new element: x_axis but rounded to 10 decimal places
        neurons = np.zeros((n, len(x_axis))) # n by len(x_axis) matrix of zeros

        for i in range(n):

            loc = (max_ - min_) * (i /(n-1)) + min_
            neurons[i] = norm.pdf(x_axis, loc, s[t])
            neurons[i] = neurons[i] / np.max(neurons[i])

        neurons_list.append(neurons)
        t += 1

    return neurons_list, x_axis_list


# Select the parameters for uniform coverage of the range of possible values of each feature by Gaussians and apply the function written above:
sigm = [0.1, 0.1, 0.2, 0.1]
d = Gaus_neuron(df_, 10, 0.001, sigm)


# Now visualize Gaussians for our dataset for each input feature:
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)

fig.set_figheight(8)
fig.set_figwidth(10)

k = 0

for ax in [ax1, ax2, ax3, ax4]:

    ax.set(ylabel = f'{df_.columns[k]} \n\n Excitation of Neuron')

    for i in range(len(d[0][k])):

        ax.plot(d[1][k], d[0][k][i], label = i + 1)

    k+=1


plt.legend(title = "Presynaptic neuron number \n      in each input column" , bbox_to_anchor = (1.05, 3.25), loc = 'upper left')
plt.suptitle(' \n\n  Gaussian receptive fields for Iris dataset', fontsize = 15)
ax.set_xlabel(' Presynaptic neurons and\n input range of value feature', fontsize = 12, labelpad = 15)
plt.show()


print("\nDone! :)")

