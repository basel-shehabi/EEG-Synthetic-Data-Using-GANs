## MSc Final Year Project: Generating Synthetic EEG Data Using GANs
Side note, the requirements.txt assumes you have a CUDA enabled GPU (NVIDIA graphics card) so please edit it where appropriate in order to get the right version(s) of Tensorflow and Keras!

The load_and_process script needs to be loaded only once per subject to save time. You can then call them using np.load()
