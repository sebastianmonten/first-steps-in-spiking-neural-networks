"""
A simple impulse neural model from scratch, showing how to encode data using Gaussian receptive fields using only NumPy and a little bit of Pandas.
By Andrey Urusov
"""

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

# Print the first few lines of the dataset
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

        vol = df[col].values # numpy.ndarray of all the values in the column
        min_ = np.min(vol)
        max_ = np.max(vol)
        x_axis = np.arange(min_, max_, step) # numpy.ndarray {min_, min_ + step, min_ + 2*step, ... , max_ - step}
        x_axis[0] = min_  # hard code first element to min_
        x_axis[-1] = max_ # hard code last element to max_
        x_axis_list.append(np.round(x_axis, 10)) # x_axis_list gets a new element: x_axis but rounded to 10 decimal places
        neurons = np.zeros((n, len(x_axis))) # n by len(x_axis) matrix of zeros

        for i in range(n):

            loc = (max_ - min_) * (i /(n-1)) + min_ # loc is the ith element in [min_, min_+1*range_len/(n-1), ...,  min_+(n-2)*range_len/(n-1), max_]
            neurons[i] = norm.pdf(x_axis, loc, s[t]) # neurons[i] = [f_X(x_axis[0]), f_X(x_axis[1]), ...] , X ∈ N(loc, s[i])
            neurons[i] = neurons[i] / np.max(neurons[i]) # scale neurons[i] so that max value is 1

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
        # i = 0, 1, ..., 9
        ax.plot(d[1][k], d[0][k][i], label = i + 1)

    k+=1


plt.legend(title = "Presynaptic neuron number \n      in each input column" , bbox_to_anchor = (1.05, 3.25), loc = 'upper left')
plt.suptitle(' \n\n  Gaussian receptive fields for Iris dataset', fontsize = 15)
ax.set_xlabel(' Presynaptic neurons and\n input range of value feature', fontsize = 12, labelpad = 15)
plt.show()

# Now let’s examine the encoding logic in detail using the first five data points of the “SepalWidthCm” feature as an example:
# we will draw dotted vertical segments for the first five values of the “SepalWidthCm” feature and locate their intersections with the Gaussian presynaptic neurons. These points will be marked with red dots:
x_input = 5
fig, ax = plt.subplots(1)

fig.set_figheight(5)
fig.set_figwidth(15)

ax.set(ylabel = df_.columns[1])

for i in range(len(d[0][1])):
    ax.plot(d[1][1], d[0][1][i])

for n in range(x_input):

    plt.plot(np.tile(df_['SepalWidthCm'][n], (10,1)), 
         d[0][1][np.tile(d[1][1] == df_['SepalWidthCm'][n], (10,1))], 
                                                            'ro', markersize=4)

    plt.vlines(x = df_['SepalWidthCm'][n], ymin = - 0.1, ymax = 1.1, 
               colors = 'purple', ls = '--', lw = 1, label = df_['SepalWidthCm'][n])

    plt.text(df_['SepalWidthCm'][n] * 0.997, 1.12, n + 1, size = 10)


plt.legend(title = "First five input:", bbox_to_anchor = (1.0, 0.7), 
                                                            loc = 'upper left')

plt.suptitle('Gaussian receptive fields for Iris dataset. \n \
                A detailed description of the idea using the example of the first five value "SepalWidthCm"',
            fontsize = 15)

ax.set_xlabel('Input value X ∈ [x_min, x_max] of column', fontsize = 12, labelpad = 15)
ax.set_ylabel('Excitation of a Neuron ∈ [0,1]', fontsize = 12, labelpad = 15)

plt.show()

# Now output the numerical values of the intersection points of the input of each value with each Gaussian (presynaptic neuron):
np.set_printoptions(formatter={'float_kind':'{:f}'.format})
five_x = np.zeros((5, 10)) 

for n in range(x_input):
    # Fills each row of five_x with values from d[0][1] based on a condition that
    # checks if d[1][1] equals the corresponding value in the SepalWidthCm column of df_.
    # The condition is tiled to match the 10 columns.
    five_x[n,:] = d[0][1][np.tile(d[1][1] == df_['SepalWidthCm'][n], (10,1))]

print(five_x)

# Let’s find the latency of each presynaptic neuron as 1 — (excitation of the presynaptic neuron),
# provided that the excitation of the neuron is greater than 0.1, otherwise we consider the presynaptic neuron to be unexcited at this iteration:
five_x = np.where(five_x > 0.1, 1 - five_x, np.nan)
five_x[five_x == 0] = 0.0001
print(five_x)


# Now let’s visualize the process of spike occurrence taking into account the latency.
# Black dots on the graph — the moment the spike is emitted by the presynaptic neuron:
fig, ax = plt.subplots(5, figsize=(10, 8))

for i in range(5):
    ax[i].scatter(x = five_x[i], y = np.arange(1, 10 + 1), s = 10, color = 'black')
    ax[i].hlines(xmin = 0, xmax=1, y=np.arange(1, 11, 1),
               colors = 'purple', ls = '--', lw = 0.25)
    ax[i].yaxis.set_ticks(np.arange(0, 11, 1))
    ax[i].set_ylabel(f'x{i+1} = {df_.iloc[i,1]}\n (period {i+1}) \n\n № \npresynaptic neuron',
                                                                  fontsize = 7)
    ax[i].set_xlim(0, 1)
    ax[i].set_ylim(0, 10 * 1.05)
    ax[i].tick_params(labelsize = 7)

ax[i].set_xlabel('Spike Latancy')
plt.suptitle(' \n\nInput after applying latancy coding \nusing the Gaussian receptive fields method', 
                                                                 fontsize = 12)
plt.show()



# Now we will do this for all values of all features:
def Lat_Spike(df, d, n):

    for i in range(len(df.columns)):
        # i = 0, 1, 2, 3

        k = len(df.iloc[:, i]) # len of each column, 150
        st1 = np.tile(d[1][i], (k, 1)) # st1 is an array of the x_axis for column i, repeated 150 times
        st2 = df.iloc[:, i].values.reshape(-1, 1) # st2 is the ith column reshaped into a 2D array with one column and as many rows as needed.
        ind = (st1 == st2)
        exc = np.tile(d[0][i], (k, 1)).reshape(k, n, len(d[0][i][0]))[
            np.repeat(ind, n, axis=0).reshape(k, n, len(ind[0]))].reshape(k, n)
        lat_neuron = np.transpose(np.where(exc > 0.1, 1 - exc, np.nan))

        if i == 0:
            lat_neuron_total = lat_neuron
        else:
            lat_neuron_total = np.concatenate((lat_neuron_total, lat_neuron), axis = 0)

    lat_neuron_total[lat_neuron_total == 0] = 0.0001

    return lat_neuron_total

fin = Lat_Spike(df_, d, 10)



# Visualize the moments of spike generation by presynaptic neurons for the first value of each of the four features:
fig, ax = plt.subplots(4, figsize=(10, 6))

for i in range(4):

    ax[i].scatter(x = fin[i * 10:10 * (1 + i), 0], y = np.arange(1, 10 + 1), s = 10, color = 'r')
    ax[i].hlines(xmin = 0, xmax = 1, y=np.arange(1, 11, 1), 
               colors = 'purple', ls = '--', lw = 0.25)
    ax[i].yaxis.set_ticks(np.arange(0, 11, 1))
    ax[i].set_ylabel(f'col_{i + 1}: {(df_.columns)[i]} \n x1 = {df_.iloc[0, i]} \n (period {1})\n\n № \npresynaptic neuro', fontsize = 6)
    ax[i].set_xlim(0, 1)
    ax[i].set_ylim(0, 10 * 1.05)
    ax[i].tick_params(labelsize = 7)

ax[i].set_xlabel('Spike Latancy')
plt.suptitle('\nFirst input in each column \n after applying latancy coding using the Gaussian receptive fields method', fontsize = 10)
plt.show()


# Now present the results in the form of a DataFrame.
# By rows — presynaptic neurons,
# by columns — the number of the input data set:

Final_df = pd.DataFrame(fin)
print(Final_df)



# Let’s visualize the obtained latencies for the first three learning periods,
# considering one period to be equal to 10 ms, therefore, we will scale the latency of
# each presynaptic neuron (total time interval by three periods is 30 ms):
fig, ax = plt.subplots(1, figsize=(10, 6))
h = 3

for i in range(h):
    ax.scatter(x = (i+Final_df.iloc[:,i].values)*10, y = np.arange(1, 41), s = 6, color = 'black')

    plt.vlines(x = (i) * 10, ymin = 0, ymax = 40, 
               colors = 'purple', ls = '--', lw = 0.5)
    ax.tick_params(labelsize = 7)

ax.yaxis.set_ticks(np.arange(1, 41, 1))
ax.xaxis.set_ticks(np.arange(0, (h+1)*10, 10))
ax.set_xlabel('time (ms)')
ax.set_ylabel('№ presynaptic neuron')
plt.suptitle(' \n\nSpikes of presynaptic neurons for first 30 ms', fontsize = 10)
plt.gca().invert_yaxis()
plt.show()



####################################################################################
# LIF neuron model
# Part one. Subsample size of 60: the first 20 values for each flower type.
####################################################################################

def model_data(ind, ind_type, lat_ne, start, end):

    """
    The data in the original data set are distributed sequentially by iris types:
    a total of 150 data sets of which the first 50 records belong to Iris-setosa,
    50–100 — Iris-versicolor, 100–150 — Iris-virginica.
    
    This is a function that will select the data set we need from the portfolio,
    in which there will be an equal number of instances of each type.

    Args:
        ind:        
        ind_type:   
        lat_ne:     
        start:      
        end:

    Returns:
        
    """ 
    train_stack = np.vstack((lat_ne[ind_type[ind, 0] + start:ind_type[ind, 0] + end],
                            lat_ne[ind_type[ind, 1] + start:ind_type[ind, 1] + end],
                            lat_ne[ind_type[ind, 2] + start:ind_type[ind, 2] + end]))
    train_stack = np.where(train_stack > 0, train_stack, 0)
    
    return train_stack


# we need to somehow increase the weights of active synapses,
# forming a set of weights foreach postsynaptic neuron for further
# correction using STDP (Spike-timing-dependent plasticity) at the second stage.

lat_ne = np.transpose(Final_df.values)
ind_type = np.array(([0, 50, 100], [50, 100, 0], [100, 0, 50]))
list_weight = np.zeros((3,40))

for ind in range(3):
    
    train_stack = model_data(ind, ind_type, lat_ne, 0, 20)
    tr_ar = np.where(np.transpose(train_stack) > 0, 2 * (1 - np.transpose(train_stack)), 0)
    tr_ar[:, 20:] = tr_ar[:, 20:] * (-1)
    tr_ar = pd.DataFrame(tr_ar)
    tr_ar[20] = tr_ar.iloc[:,:20].sum(axis = 1) + 0.1
    tst_ar = np.float64(np.transpose(np.array(tr_ar.iloc[:,20:])))
    
    for i in range(1, len(tst_ar)):
        
        tst_ar[0][((np.round(tst_ar[0], 4) > 0.1) & (tst_ar[i] == 0))] += - np.float64(
            np.sum(tst_ar[i][np.round(tst_ar[0], 4) > 0.1]) / len(tst_ar[0][((
                np.round(tst_ar[0], 4) > 0.1) & (tst_ar[i] == 0))]))
        tst_ar[0][np.round(tst_ar[0], 4) > 0.1] += tst_ar[i][np.round(tst_ar[0], 4) > 0.1]
        tst_ar[0][tst_ar[0] < 0.1] = 0.1
        
    list_weight[ind, :] = tst_ar[0]
print(list_weight)














# Changing the membrane potantial of a postsynaptic neuron

def LIF_SNN(n, l, data, weight, v_spike):
    
    V_min = 0
    V_spike = v_spike
    r = 5
    tau = 2.5
    dt = 0.01
    t_max = 10
    time_stamps = t_max / dt
    time_relax = 10
    v = np.zeros((n, l, int(time_stamps)))
    t_post = np.zeros((n, l))
    t_post_ = np.zeros((n, int(l / 3)))
    v[:, :, 0] = V_min
    
    for n in range(n):
        for u in range(l):
            
            t = 0
            f0 = (np.round(data[u][np.newaxis].T, 3) * 1000).astype(int)
            f1 = np.tile(np.arange(1000), (40, 1))
            f2 = np.where(((f1 == f0) & (f0 > 0)), 1, 0)
            f2 = f2 * weight[n][np.newaxis].T
            spike_list = np.sum(f2.copy(), axis = 0)

            for step in range(int(time_stamps) - 1):
                if v[n, u, step] > V_spike:
                    t_post[n, u] = step
                    v[n, u, step] = 0
                    t = time_relax / dt
                elif t > 0:
                    v[n, u, step] = 0
                    t = t - 1

                v[n, u, step + 1] = v[n, u, step] + dt / tau * (-v[n, u, step] + r * spike_list[step])
        t_post_[n, :] = t_post[n, n * int(l / 3):n * int(l / 3) + int(l / 3)]
    
    return v, t_post_, t_post






# Function for visualizing spike moments of postsynaptic neurons:
def spike_plot(spike_times, one_per, n, cur_type):
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize = (25, 10))#, dpi = 70)
    
    if one_per:
        k, t, a  = 1, n, 0
        cur = cur_type
    else:
        k, t, a = len(spike_times[0]), 0, 1
        cur = 1
        
    spike_times[spike_times == 0] = np.nan
    di = {0: 'blue', 1: 'red', 2: 'black'}
    di_t = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}
    p = 0
    
    for ax in [ax1, ax2, ax3]:
        for i in range(k * t, k + t):
            ax.vlines(x = spike_times[p, i] / 100 + i * a * 10, ymin = 0.0, ymax = 1.1, 
                       colors = di[p], ls = '-', lw = 3)
            ax.set_ylabel(f'Neuron {p + 1} \n {di_t[p]}', fontsize = 15)
            
        if one_per:
            ax.axvspan(0, int(k * 10), color = di[cur - 1], alpha = 0.05, label = di_t[cur - 1])
            ax.margins(0)
        else:
            ax.axvspan(0, int(k * 10 / 3), color = di[0], alpha = 0.05, label = di_t[0])
            ax.axvspan(int(k * 10 / 3), int(k * 10 * 2 / 3), color = di[1], alpha = 0.05, label = di_t[1])
            ax.axvspan(int(k * 10 * 2 / 3), int(k * 10 * 3 / 3), color = di[2], alpha = 0.05, label = di_t[2])
            ax.set_xlim(0, k * 10)
            ax.margins(0)
            
        p += 1
        
    
    if one_per:
        plt.suptitle(f' \n\n Moment of spike of postsynaptic neurons for train period {n}', fontsize = 20)
        plt.legend(title = "    Part of a type set:" ,bbox_to_anchor = (1, 1.9), loc = 'upper left',
               fontsize = 15, title_fontsize = 15)
    else:
        plt.suptitle(f' \n\n Moment of spike of postsynaptic neurons on the used part of the dataset', fontsize = 20)
        plt.legend(title = "    Part of a type set:" ,bbox_to_anchor = (1, 2.1), loc = 'upper left',
               fontsize = 15, title_fontsize = 15)
    
    plt.xlabel('Time (ms)', fontsize = 15)
    plt.show()





# Function for visualizing membrane potential of each postsynaptic neuron:

def v_plot(v):
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize = (25, 10))#, dpi = 70)
    k = len(v[0,:,:])
    di = {0: 'blue', 1: 'red', 2: 'black'}
    di_t = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}
    p = 0
    
    for ax in [ax1, ax2, ax3]:
        for i in range(k):
            ax.plot(np.arange(i * 10, (i + 1) * 10, 0.01), v[p, i, :], di[p], linewidth = 1)
            ax.set_ylabel(f' Neuron {p + 1} \n {di_t[p]} \nV (mV)', fontsize = 15)

        ax.axvspan(0, int(k * 10 / 3), color = di[0], alpha = 0.05, label = di_t[0])
        ax.axvspan(int(k * 10 / 3), int(k * 10 * 2 / 3), color = di[1], alpha = 0.05, label = di_t[1])
        ax.axvspan(int(k * 10 * 2 / 3), int(k * 10 * 3 / 3), color = di[2], alpha = 0.05, label = di_t[2])
        ax.margins(0)

        p += 1
    
    plt.legend(title = "    Part of a type set:" ,bbox_to_anchor = (1, 2), loc = 'upper left', fontsize = 15, title_fontsize = 15)
    plt.xlabel('Time (ms)', fontsize = 15)
    plt.suptitle(' \n Activity of postsynaptic neurons on the used part of the dataset \n (Membrane potential)', fontsize = 20)
    plt.show()







# Accuracy function. If multiple postsynaptic neurons generate spikes during one period, the postsynaptic neuron that generated the spike first is considered to have fired:
def accuracy_snn(spike_time, start, end, df, ind_type, ind):
    
    type_dict = {'Iris-setosa': 1, 'Iris-versicolor': 2, 'Iris-virginica': 3}
    target_type_total = np.array(df.replace({'Species': type_dict}).iloc[:, - 1])
    target_type = np.vstack((target_type_total[ind_type[ind, 0] + start:ind_type[ind, 0] + end],
                            target_type_total[ind_type[ind, 1] + start:ind_type[ind, 1] + end],
                            target_type_total[ind_type[ind, 2] + start:ind_type[ind, 2] + end])).flatten()
    
    spike_time_ = np.where(spike_time > 0, np.array(([1], [2], [3])), np.nan)
    final_test = np.full([len(spike_time[0])], np.nan).astype(int)
    for i in range(len(spike_time[0])):
        try:
            final_test[i] = spike_time_[:, i][spike_time[:, i] == np.min(spike_time[:, i][spike_time[:, i] > 0])][0]
        except:
            final_test[i] = 0
    
    ac = np.sum(np.where(final_test == target_type, 1, 0)) / len(target_type)

    return final_test, target_type, print('accur.:', np.round(ac * 100, 2), '%')










# We adjusted and increased the weights on the first 20 instances of each type for each postsynaptic neuron,
# resulting in three sets of weights. Let’s examine the membrane potential profile of each postsynaptic neuron
# with these obtained weights on the same first part of the training set. At this stage, we will not limit
# the membrane potential to a threshold level, choosing it to be equal to 100:
train_stack = model_data(0, ind_type, lat_ne, 0, 20)
res = LIF_SNN(3, 60, train_stack, list_weight, 100)
v = res[0]

v_plot(v)




# Overall, it looks good, with each postsynaptic neuron’s activity area clearly visible.
# The membrane potential profile of the first neuron looks the best,
# while neurons 2 and 3 are more responsive to “foreign” spikes that should not significantly
# change their potentials — this could lead to incorrect classification.
# Let’s look at the spike times and accuracy at this stage with a threshold voltage value of 0.25:
res = LIF_SNN(3, 60, train_stack, list_weight, 0.25)
spike_time = res[2]
spike_plot(spike_time, False, False, False)
accuracy_snn(spike_time, 0, 20, df, ind_type, 0)[2]


# The accuracy is real good! Let’s examine a few periods where one of the postsynaptic neurons has false activations.
# We’ll try to understand what’s happening and how it affects accuracy.
# Let’s look at the last false spike of the first postsynaptic neuron,
# which occurs in period 46 of the first part oftraining:
spike_plot(spike_time, True, 46, 3)


# Let’s examine the second false spike of the third postsynaptic neuron,
# which occurs in period 24 of the first part of training:
spike_plot(spike_time, True, 24, 2)





####################################################################################
# Part two. Subsample size of 60: the second 20 values for each flower type.
####################################################################################

# At this stage, we are training on the next set of input data using local STDP learning.
# Before we proceed, let’s see what the result and accuracy would be if we applied the
# current weights to the second training set:
train_stack = model_data(0, ind_type, lat_ne, 20, 40)
res = LIF_SNN(3, 60, train_stack, list_weight, 100)
v = res[0]

v_plot(v)
res = LIF_SNN(3, 60, train_stack, list_weight, 0.25)
spike_time = res[2]
spike_plot(spike_time, False, False, False)
accuracy_snn(spike_time, 20, 40, df, ind_type, 0)[2]




# Applying STDP
# We will calculate the weight change for each presynaptic neuron for each postsynaptic neuron

res = LIF_SNN(3, 60, train_stack, list_weight, 0.25)
t_post = res[1]
A_p = 0.8
A_m = A_p * 1.1

for n in range(3):
    for u in range(20):
        
        t1 = np.round(train_stack[u + 10 * n] * 1000)
        t2 = t1.copy()
        
        t2[((t1 <= t_post[n, u]) & (t1 > 0))] = A_p * np.exp((t1[((t1 <= t_post[n, u]) & (t1 > 0))] - t_post[n, u]) / 1000)
        t2[((t1 > t_post[n, u]) & (t1 > 0))] = - A_m * np.exp((t_post[n, u] - t1[((t1 > t_post[n, u]) & (t1 > 0))]) / 1000)
        
        list_weight[n, :] += t2
        
list_weight[list_weight < 0] = 0
print(list_weight)


# We have adjusted the weights, let’s see how the accuracy of the model has now changed on the second set of training instances:
res = LIF_SNN(3, 60, train_stack, list_weight, 0.25)
spike_time = res[2]
spike_plot(spike_time, False, False, False)
accuracy_snn(spike_time, 20, 40, df, ind_type, 0)[2]






#######################################################################################################################
# Part three.
# Now let’s check the classification accuracy on the entire training set
# (all first 40 instances of each class) using by this weights:
#######################################################################################################################

train_stack = model_data(0, ind_type, lat_ne, 0, 40)
res = LIF_SNN(3, 120, train_stack, list_weight, 100)
v = res[0]

v_plot(v)
res = LIF_SNN(3, 120, train_stack, list_weight, 0.25)
spike_time = res[2]
spike_plot(spike_time, False, False, False)
accuracy_snn(spike_time, 0, 40, df, ind_type, 0)[2]




#######################################################################################################################
# TESTING, Part four.
# Now let’s run the network on the test set (the last 10 instances of each class):
#######################################################################################################################

train_stack = model_data(0, ind_type, lat_ne, 40, 50)
res = LIF_SNN(3, 30, train_stack, list_weight, 100)
v = res[0]
res = LIF_SNN(3, 30, train_stack, list_weight, 0.25)
spike_time = res[2]

v_plot(v)
spike_plot(spike_time, False, False, False)
accuracy_snn(spike_time, 40, 50, df, ind_type, 0)[2]

# The accuracy seems to be accur.: 100.0 %
# There seems to be one false firing of the second postsynaptic neuron in period 27, let’s examine it more closely:
spike_plot(spike_time, True, 27, 3)








print("\nDone! :)")

