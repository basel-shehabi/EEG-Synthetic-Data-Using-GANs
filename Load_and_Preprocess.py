import os
import mne
import numpy as np
import matplotlib.pyplot as plt
from mne import time_frequency
from sklearn.model_selection import train_test_split


class LoadData:
    def __init__(self, subject_number=1):
        self.subject_number = subject_number
        self.left, self.right, self.bimanual = self.get_crop_filtered_data_from_fif()

    def get_crop_filtered_data_from_fif(self, tmin=0, tmax=4):

        '''
        Gets data from a .fif file, equalises the event/task count,
        filters the data from 5Hz to 30Hz Bandpass, crops it from a 0-4 sec window
        and returns the trials from a certain task along side its labels in the form
        of a tuple (e.g. (left subject data(trials x channels x samples, left labels)
        Useful for debugging or knowing certain info regarding data
        :return:
        '''

        # Load data from current working directory. File must be saved in a folder as 'S + Subject number'
        # Also splits and equalises data in order to get equal trials per task/label
        sample_dir = os.getcwd()
        epochs = mne.read_epochs(sample_dir + '\\' 'S%s\\epochs_epo.fif' % self.subject_number, preload=True)
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
        :return:
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
        Norm_Left_MEpoch = np.swapaxes(np.swapaxes(Left_MEpoch, 1, 3), 1, 2)
        Norm_Right_MEpoch = np.swapaxes(np.swapaxes(Right_MEpoch, 1, 3), 1, 2)
        Norm_Biman_MEpoch = np.swapaxes(np.swapaxes(Biman_MEpoch, 1, 3), 1, 2)

        # ... then normalise the data for faster training
        Norm_Left_MEpoch = 2 * (Norm_Left_MEpoch - np.min(Norm_Left_MEpoch, axis=0)) / \
                           (np.max(Norm_Left_MEpoch, axis=0) - np.min(Norm_Left_MEpoch, axis=0)) - 1
        Norm_Right_MEpoch = 2 * (Norm_Right_MEpoch - np.min(Norm_Right_MEpoch, axis=0)) / \
                            (np.max(Norm_Right_MEpoch, axis=0) - np.min(Norm_Right_MEpoch, axis=0)) - 1
        Norm_Biman_MEpoch = 2 * (Norm_Biman_MEpoch - np.min(Norm_Biman_MEpoch, axis=0)) / \
                            (np.max(Norm_Biman_MEpoch, axis=0) - np.min(Norm_Biman_MEpoch, axis=0)) - 1

        return (Norm_Left_MEpoch, left_label), (Norm_Right_MEpoch, right_label), (Norm_Biman_MEpoch, bimanual_label)


def data_split_save(Input_Dataset, task_number=1, subject_number=1):

    '''
    Splits the data for a given subject per task for a training dataset and a testing dataset
    to feed into the model when appropriate. Returns None to the caller but
    returns X_train[task], y_train[task], X_test[task], y_test[task]
    and then saves these respectively  to disk as two '.npz' files
    (first for training, the other for testing) so that you don't have to
    run this module over and over again when reusing data from the same subject.
    The split used is 75/25 which is the default used by sklearn.
    '''

    Train_task = Input_Dataset[0]
    Train_task_label = Input_Dataset[1]

    X_train, y_train, X_test, y_test = train_test_split(Train_task, Train_task_label)

    np.savez('Subject{}Train{}.npz'.format(subject_number, task_number), x=X_train, y=y_train)
    np.savez('Subject{}Test{}.npz'.format(subject_number, task_number), x=X_test, y=y_test)

    return None


# Instantiate Object
x = LoadData()
y, z, k = x.get_normalized_cwt_data()

# Split and save the data
input = [(y, 1), (z, 2), (k, 3)]
for obj, idx in input:
    data_split_save(obj, task_number=idx)

# Sample plot to make sure things work!
t = np.array(y[0])
print(t.shape)

plt.figure(1)
plt.title('C4 Spectrogram')
plt.imshow(t[25, :, :, 2], aspect='auto')
plt.colorbar()
plt.show()
