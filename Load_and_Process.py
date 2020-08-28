import os
import numpy as np
import matplotlib.pyplot as plt
from mne import time_frequency, read_epochs
from sklearn.model_selection import train_test_split


class LoadData:
    def __init__(self, subject_number=1):
        self.subject_number = subject_number
        self.left, self.right, self.bimanual = self.get_crop_filtered_data_from_fif()

    def get_epoch_data(self, channels=(7,9,11)):
        '''
        Gets subject data from a .fif file, equalises the event/task count and returns the
        overall trial data for the subject. Used for visualisation or feeding into the
        standard GAN that accepts the overall trial data per subject thats unfiltered
        and uncropped.
        :return: (Trial array, Label data per trial array)
        '''

        # Load data from current working directory. File must be saved in a folder as 'S + Subject number'
        project_dir = os.getcwd()
        epochs = read_epochs(project_dir + '\\' 'S%s\\epochs_epo.fif' % self.subject_number, preload=True)
        epochs.equalize_event_counts(['Left', 'Right', 'Bimanual'])
        EEGarray = epochs.get_data()
        labels = epochs.events[:, 2]

        # Select what channels we want
        # And formats the array in the form of (Trials, Time Samples, Channels) for the GAN
        EEGarray = np.swapaxes(EEGarray[:, channels, :], 1, 2)

        print(EEGarray.shape)


        return (EEGarray, labels)

    def get_crop_filtered_data_from_fif(self, tmin=0, tmax=4):
        '''
        Gets subject data from a .fif file, equalises the event/task count,
        filters the data from 5Hz to 30Hz Bandpass, crops it from a 0-4 sec window
        and returns the trials from a certain task along side its labels in the form
        of a tuple (e.g. (left subject data(trials x channels x samples, left labels)
        Useful for debugging or knowing certain info regarding data
        :return: (Task Data, Task Label Array)
        '''

        # Load data from current working directory. File must be saved in a folder as 'S + Subject number'
        # This time we also split and equalise the data in order to get equal trials per task/label
        project_dir = os.getcwd()
        epochs = read_epochs(project_dir + '\\' 'S%s\\epochs_epo.fif' % self.subject_number, preload=True)
        epochs.equalize_event_counts(['Left', 'Right', 'Bimanual'])

        # Do the filtering and cropping to capture the prominent features
        epochs = epochs.filter(5, 30, fir_design='firwin')
        epochs_filter = epochs.copy().crop(tmin=tmin, tmax=tmax)

        left_epochs = epochs_filter['Left']
        right_epochs = epochs_filter['Right']
        bimanual_epochs = epochs_filter['Bimanual']

        left = left_epochs.get_data()
        right = right_epochs.get_data()
        bimanual = bimanual_epochs.get_data()

        # Get labels
        left_labels = left_epochs.events[:, 2]
        right_labels = right_epochs.events[:, 2]
        bimanual_labels = bimanual_epochs.events[:, 2]

        return (left, left_labels), (right, right_labels), (bimanual, bimanual_labels)

    def get_normalized_cwt_data(self, channels=(7, 9, 11)):
        '''
        Applies Morlet Continuous Wavelet Transform for filter extraction on the Epoch data from a
        certain channel range e.g. (C3, CPz, C4), normalizes it and then returns the data in a tuple
        in the form of (Trial, Freq Sample, Time Sample, Channel)
        alongside its respective labels (i.e. (Left_MEpoch, Left Labels)). This is done to reduce the
        computational load during training time.
        NOTE: the absolute value is taken to remove imaginary components
        :return: (Normalized, CWT Task Data, Task Label Array)
        '''

        if channels is None:
            channels = [7, 9, 11]

        left_filtered, right_filtered, bimanual_filtered = self.get_crop_filtered_data_from_fif()

        left_filtered, left_label = left_filtered[0], left_filtered[1]
        left_filtered = left_filtered[:, channels, :]

        right_filtered, right_label = right_filtered[0], right_filtered[1]
        right_filtered = right_filtered[:, channels, :]

        bimanual_filtered, bimanual_label = bimanual_filtered[0], bimanual_filtered[1]
        bimanual_filtered = bimanual_filtered[:, channels, :]


        freqs = np.logspace(*np.log10([5, 30]), num=25)
        n_cycles = freqs / 2.
        sfreq = 256

        # Perform a Morlet CWT on each epoch for feature extraction
        Left_MEpoch = time_frequency.tfr_array_morlet(left_filtered, sfreq=sfreq, freqs=freqs, n_cycles=n_cycles,
                                                      use_fft=True, decim=3, output='complex', n_jobs=1)

        Right_MEpoch = time_frequency.tfr_array_morlet(right_filtered, sfreq=sfreq, freqs=freqs, n_cycles=n_cycles,
                                                       use_fft=True, decim=3, output='complex', n_jobs=1)

        Biman_MEpoch = time_frequency.tfr_array_morlet(bimanual_filtered, sfreq=sfreq, freqs=freqs, n_cycles=n_cycles,
                                                       use_fft=True, decim=3, output='complex', n_jobs=1)

        Left_MEpoch, Right_MEpoch, Biman_MEpoch = np.abs(Left_MEpoch), np.abs(Right_MEpoch), np.abs(Biman_MEpoch)

        # Swap the axes to feed into GAN models later
        Left_MEpoch = np.swapaxes(np.swapaxes(Left_MEpoch, 1, 3), 1, 2)
        Right_MEpoch = np.swapaxes(np.swapaxes(Right_MEpoch, 1, 3), 1, 2)
        Biman_MEpoch = np.swapaxes(np.swapaxes(Biman_MEpoch, 1, 3), 1, 2)

        # ... then normalise the data for faster training
        Norm_Left_MEpoch = 2 * (Left_MEpoch - np.min(Left_MEpoch, axis=0)) / \
                           (np.max(Left_MEpoch, axis=0) - np.min(Left_MEpoch, axis=0)) - 1
        Norm_Right_MEpoch = 2 * (Right_MEpoch - np.min(Right_MEpoch, axis=0)) / \
                            (np.max(Right_MEpoch, axis=0) - np.min(Right_MEpoch, axis=0)) - 1
        Norm_Biman_MEpoch = 2 * (Biman_MEpoch - np.min(Biman_MEpoch, axis=0)) / \
                            (np.max(Biman_MEpoch, axis=0) - np.min(Biman_MEpoch, axis=0)) - 1

        return (Norm_Left_MEpoch, left_label), (Norm_Right_MEpoch, right_label), (Norm_Biman_MEpoch, bimanual_label)

def data_save(Input_Dataset, subject_number=1):
    '''
    Saves the channel specific epoch data into a .npz file for a specified subject.
    Used to load data into the standard GAN
    :param Input_Dataset:
    :param subject_number:
    :return: None
    '''
    Train_data = Input_Dataset[0]
    Train_labels = Input_Dataset[1]

    np.savez('Subject{}Data.npz'.format(subject_number), x=Train_data, y=Train_labels)

    return None

def data_split_save(Input_Dataset, task_number=1, subject_number=1):
    '''
    Splits the data for a given subject per task for a training dataset and a testing dataset
    to feed into the model when appropriate. Returns None to the caller but
    fetches X_train[task], y_train[task], X_test[task], y_test[task]
    and then saves these respectively to disk as two '.npz' files
    (first for training, the other for testing) so that you don't have to
    run this module over and over again when reusing data from the same subject.
    The split used is 75/25 which is the default used by sklearn.
    :return: None
    '''

    Train_task = Input_Dataset[0]
    Train_task_label = Input_Dataset[1]

    X_train, X_test, y_train, y_test = train_test_split(Train_task, Train_task_label)

    np.savez('Subject{}Train{}.npz'.format(subject_number, task_number), x= X_train, y= y_train)
    np.savez('Subject{}Test{}.npz'.format(subject_number, task_number), x= X_test, y= y_test)

    return None


# Instantiate Object
x = LoadData(subject_number=1)
y, z, k = x.get_normalized_cwt_data(channels=(9,))
r = x.get_epoch_data(channels=(0, 7, 9, 11, 19))

data_save(r)

p = np.array(r[0])
print(p.shape)

t = np.array(y[0])
print(t.shape)

# Split and save the cropped and normalised data
input = [(y, 1), (z, 2), (k, 3)]
for obj, idx in input:
    data_split_save(obj, task_number=idx)

# Sample plots to make sure things work!
# Data of Interest:
trial = 248
trial_cropped = 30 # Different than above since this is for the cropped and re-filtered data
subject = 1
channel = 3

# First plot is an EEG time series data for one trial
EEG_Time_Series = (p[trial, :, channel]) # EEG Data for one trial, one channel (All time samples)
print(EEG_Time_Series.shape)
plt.plot(EEG_Time_Series)  # Plots the 248th trial on the axis
plt.title('Trial {} from selected channel {}, C4 for subject {}'.format(trial, channel, subject))
plt.xlabel('Time Stamp')
plt.ylabel('Voltage Reading in Microvolts')
plt.show()

# Next one is an image plot, where each row/point on the y-axis is a trial vs. time stamp
plt.imshow(p[:, :, channel], aspect='auto')
plt.title('Subject {} trials from selected channel {}, C4'.format(subject, channel))
plt.xlabel('Time Stamp')
plt.ylabel('Trial/Epoch Number')
plt.show()

# Last one is a spectrogram for the time-frequency data for the left task
plt.imshow(t[trial_cropped, :, :, 0], aspect='auto')
plt.title('Normalised Spectrogram for subject {}, trial {} and selected channel {}/C4'
          .format(subject, trial_cropped, channel))
plt.xlabel('Time Stamp')
plt.ylabel('Frequency Stamp')
plt.colorbar()
plt.show()