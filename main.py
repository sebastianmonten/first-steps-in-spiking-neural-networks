import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import warnings
warnings.filterwarnings("ignore")

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


'''
    Function that generates 10 Gaussians for each input feature so that:

    - means of each Gaussian are evenly distributed between the extreme values
    of the range, including the boundaries for each feature 
    ( “Sepal Length”, “Sepal Width”, “Petal Length”, and “Petal Width”)

    - the height of each Gaussian is equal to 1 is the maximum excitation
    value of the presynaptic neuron, from which late we will calculate the
    spike generation latency by presynaptic neuron.
'''
def Gaus_neuron(df, n, step, s):

    neurons_list = list()
    x_axis_list = list()
    t = 0

    for col in df.columns:

        vol = df[col].values # numpy.ndarray
        min_ = np.min(vol)
        max_ = np.max(vol)
        x_axis = np.arange(min_, max_, step) # numpy.ndarray
        x_axis[0] = min_
        x_axis[-1] = max_
        x_axis_list.append(np.round(x_axis, 10))
        neurons = np.zeros((n, len(x_axis)))

        for i in range(n):

            loc = (max_ - min_) * (i /(n-1)) + min_
            neurons[i] = norm.pdf(x_axis, loc, s[t])
            neurons[i] = neurons[i] / np.max(neurons[i])

        neurons_list.append(neurons)
        t += 1

    return neurons_list, x_axis_list

sigm = [0.1, 0.1, 0.2, 0.1]
d = Gaus_neuron(df_, 10, 0.001, sigm)


print("My tests:")
print("type(df_['SepalLengthCm'].values): ", type(df_['SepalLengthCm'].values))
print("\nDone! :)")

