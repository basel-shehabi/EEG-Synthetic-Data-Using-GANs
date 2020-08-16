import mne
import os
import numpy as np


subject = 1 # Subject number
filedir = 'D:/Ciaran Python Data/Transfer-SARzWXT6VsetANd8/motorImagery/' # Where is the data stored?  motorImagery
epochs = mne.read_epochs(filedir + 'S%s//epochs_epo.fif' %(subject) ,preload=True)

# print(epochs[:])
print(epochs.info)

# Takes an equal number of trials for each condition for a fair comparison
epochs.equalize_event_counts(['Left', 'Right', 'Bimanual'])
left_epochs = epochs['Left']
right_epochs = epochs['Right']
bimanual_epochs = epochs['Bimanual']

# epochs.plot_image(picks=['CPZ'])

EEGarray = epochs.get_data()
labels = epochs.events[:,2]
labeled_data = epochs.events[:]
id = epochs.event_id
# print(id)
# print(labels)
# print(len(labels))

print(EEGarray.shape)

label_time = []
for i in range(len(labeled_data)):
    x = labeled_data[i][0] // 256
    label_time.append(x)

# print(label_time)

# This is a recording session of roughly 54 minutes
# Data points happen every 10 seconds
epochs = epochs[['Left','Right','Bimanual']]
epochs = epochs.filter(4, 30, fir_design='firwin')
epochs_train = epochs.copy().crop(tmin= 1, tmax= 4)
epochs_data_train = epochs_train.get_data()
labels = epochs.events[:,-1]
new_labels = epochs_train.events[:]
print(labels.shape)


freqs = np.logspace(*np.log10([5, 30]), num=25)
n_cycles = freqs / 2.  # different number of cycle per frequency
MorletEpoch = mne.time_frequency.tfr_array_morlet(epochs_data_train, sfreq=256, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                                                  decim=3, output='complex', n_jobs=1)

MorletEpoch = np.abs(MorletEpoch)
print(MorletEpoch.shape)

X_train_cwt = np.swapaxes(np.swapaxes(MorletEpoch,1,3),1,2)
# scale between -1 and 1 for output of Generator
X_train_cwt_norm = 2 * (X_train_cwt - np.min(X_train_cwt,axis=0) ) / (np.max(X_train_cwt,axis=0) - np.min(X_train_cwt,axis=0)) - 1

print(X_train_cwt)
print(X_train_cwt_norm)
print(X_train_cwt_norm.shape)

print(epochs_train.info)
print(epochs_data_train.shape[0])

idx = np.random.randint(0, epochs_data_train.shape[0], 50)
signals = epochs_data_train[idx]
print(signals)
print(len(signals))

#print(new_labels)

new_label_time = []
for i in range(len(new_labels)):
    y = new_labels[i][0] // 256
    new_label_time.append(y)

print(new_label_time)
print('X_cwt_norm = {}'.format(X_train_cwt_norm.shape))

# print(labels)

'''
def get_trials(channels=1):
    trials_list = []
    for t in range(249):
        trials_list.append(MorletEpoch[t][channels][:][:])
    return trials_list


def get_classes_and_trials(channels=[7, 9, 11, 13]):
    class_list = []
    trials_lists = []
    for cc in range(len(channels)):
        class_list.append(labels.copy())
    for c in channels:
        temp_list = get_trials(channels=c)
        trials_lists.append(temp_list)
    return trials_lists, class_list


t, c = get_classes_and_trials()
t = np.array(t)

print(t.shape)
'''

x = X_train_cwt_norm
print(x.shape)
y = labels

np.savez('S1Processeds.npz', x=X_train_cwt_norm, y=labels)
