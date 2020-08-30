# MSc Final Year Project: Creating Artificial/Synthetic Brain Signal 'Encephalographic' (EEG) Data Using Generative Adversarial Networks (GANs)
![](https://img.shields.io/badge/License-MIT-blue.svg) ![Generic badge](https://img.shields.io/badge/TensorFlow-2.3.0-Red.svg)

![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)

Welcome to the repository of my master's year dissertation/project, in partial fulfilment of the requirements for my MSc Computer Systems Eng. Degree, fully titled: **Generating Synthetic EEG Data Using Deep Learning For Use In Calibrating Brain Computer Interface Systems.** If you're curious or want to know more about my project, then feel free to read along.  

A short summary of the project is provided below, as it is taken from the abstract of my dissertation:

>*Brain Computer Interface Systems have seen a surge in popularity due to recent technological advancements, causing them to be used in various fields including but not limited to rehabilitation, gaming and education. These systems allow the user to interact with other devices using their brain activity as opposed to their muscles. While this technology may seem promising to those suffering from certain conditions that cause them to be paralyzed, there is still a long way before such systems become mainstream, due to multiple challenges. One such challenge is the long calibration time, where the system must get accustomed to the user’s brain signals. It requires large training data from the user which can take hours in order to gather data that is suitable. Multiple solutions have been brought forward to suppress calibration time such as adaptive classifiers or transfer learning however, more recent methods include augmenting the training data set with artificially generated Electroencephalographic (EEG) data, either using signal segmentation/recombination techniques or Deep Learning. In this project and report, the latter is investigated, where a special type of machine learning frameworks known as ’Generative Adversarial Networks’ (GANs) are utilised to produce synthetic data for augmentation in a Motor Imagery based BCI. Three types of GAN were used: A standard GAN, Deep Convolutional GAN (DCGAN) and Wasserstein GAN (WGAN). All three managed to pick up on prominent features from the original data however, only the data from the WGAN had achieved considerable results in terms of classification accuracy when augmenting the original dataset.*

### Table of Contents  
  
1. [Introduction & Background](#Introduction)  
2. [Methodology](#Methodology)
3. [Results](#Results)
4. [Discussion & Conclusion](#Discussion&Conclusion)
5. [Future Work](#FutureWork)
6. [Usage/Installation](#Usage/Installation)
8. [Acknowledgements/References](#Acknowledgements)

# Introduction & Background
Brain Computer Interface Systems or (BCIs) are computer based systems that allow humans to communicate or control other devices, without using their peripheral nerves and muscles. Instead, BCIs work by recording brain activity and using it as a method of input which ultimately relays the user’s intent. A typical BCI system consists of four components: 

1. Signal Acquisition
2. Feature Extraction
3. Feature Translation/Pattern Recognition 
4. Device Output

During Signal Acquisition, the system records brain activity either from electrical or magnetic signals generated within such as Electroencephalography/Magnetoencephalography (EEG/MEG) respectively. Feature Extraction is a process where relevant signals relating a persons intent are picked up from the analysed and recorded signal during the previous component. Broadly speaking, there are two types of signals of interest that can be 'extracted': Evoked Potentials (EPs) and Event Related Synchronisation/Desynchronisation (ERS/ERD). EPs are signals with distinguishable peaks/amplitudes, generated in the brain after a person is subjected to an external stimulus such as a sound or moving object (referred to as Auditory Evoked Potentials/Visual Evoked Potentials), while ERS happens when certain neurons/neuronal populations exchange information/fire in sync. This is why ERS brings rise to what is known as 'Neural Oscillations' such as the Alpha, Beta and Mu frequency bands, with each band corresponding to a range of frequencies in which these neurons can fire. All of the features extracted from the signal of interest are stored in what's known as a feature vector. The next component which is Feature Translation/Pattern Recognition takes in the data from the feature vector and uses Machine Learning classification algorithms (such as Linear Discriminant Analysis or Support Vector Machines) to learn from and give an output to a device. For example, a visual EP could be used to control the movement of a character in a video game, or a P300 EP signal (an evoked potential characterised by a positive peak 300ms after simulus onset) can be used to select a letter in a spelling system. Finally, the device output component takes in the commands generated during feature translation/pattern recognition to operate the external device. The figure below shows the components in more detail. 

<p align="center">
  <img src="https://github.com/basel-shehabi/EEG-Synthetic-Data-Using-GANs/blob/master/pics/fig1.jpg">
</p>

Thus, BCIs can be seen as a promising technology that allows patients who are locked-in/paralyzed, due to suffering from a stroke or other debilitating conditions, to communicate and interact with the outside world without using their muscles. Other forms of entertainment such as Virtual Realty (VR) games are slowly starting to adopt this technology however, there are multiple challenges that need to be surmounted before this technology is more widespread. Some of these challenges include technological, neurological and ethical. Technological challenges focus mainly on issues pertaining to signal acquisition/processing methods, cost, portability and setup time. Neurological challenges include the nonstationary nature of the brain or in other words, there exists inter and intrasubject variability in the acquired signals, where signals present in one subject might not be as strong or as profound in another. This intersubject variability is attributed due to anatomical diversity in terms of brain size/area between subjects, while intrasubject variability exists due to mood variation causing a weak or unusable signal for input. Ethical challenges on the other hand deal with problems of human ethics (mind reading) and socioeconomic factors. 

Ultimately, to use the BCI system, there exists a long training and calibration time for the system to adapt to the user's signal of interest as well as to train the classifier. This training time can range anywhere from 20 minutes up until several hours depending on the type of BCI system and application. Having said that, there have been solutions put forward to reduce the training and calibration time, some of which include **Transfer Learning** (Extracting information from different areas such as raw EEG data, features, or classification domain to compensate the lack of labelled data from the test subject) or **'Adaptive' Classifiers** (classification algorithms such as Adaptive Linear Discriminant Analysis that accounts for subject variations). 

A more novel method used, with the hopes of reducing calibration time, is augmenting or adding artificial data onto the original training dataset that is used calibrate the system. This can be done using signal segmentation/recombination, where an incoming user signal is copied and segmented into smaller time frames, then rejoined with other signals that have undergone similar a similar segmentation procedure, resulting in a 'fake' or artificial signal from previously existing signals. Figure two gives a brief example of how this is done.

<p align="center">
  <img src="https://github.com/basel-shehabi/EEG-Synthetic-Data-Using-GANs/blob/master/pics/fig2.PNG">
</p>

Another method of generating these artificial signals is using Deep Learning frameworks such as **Generative Adverserial Neural Networks (GANs).** Simply put, Generative Adverserial Networks which where proposed by Goodfellow et al. is similar to the idea of game theory where two players compete against each other with only one winner. The two players in this case are neural networks where one is known as the generator and the other as the discriminator. A generator’s job is to generate artificial/fake data from a latent noise variable/random noise vector, while the discriminator continually distinguishes between which samples are generated and which are real. Therefore, the end goal for the generator is to continuously keep on generating and improving upon the artificial samples such that they match the real dataset which the discriminator is trained on, causing the discriminator to label the artificial data from the generator as real. It is also the reason why such generative networks are termed as adversarial, due to the opposing nature of both networks. Both the generator’s and discriminator’s training procedure can be expressed mathematically using a minimax problem with the equation below:


<p align="center">
  <img src="https://github.com/basel-shehabi/EEG-Synthetic-Data-Using-GANs/blob/master/pics/equation.png">
</p>


Where G, D, X and Z are generator parameters, discriminator parameters, real sample and generated sample (from noise). The function D(x) gives a probability as to whether the sample belongs to a real or generated data distribution (from P data or Pz -- the noise sampling distribution). Both the discriminator and generator parameters are trying to maximise the log (D(x)) function (discriminator labeling performance), whilst minimizing the log(1-D(G(z)). In a typical BCI setting, the framework for EEG generation is best shown in figure 3 below.

<p align="center">
  <img src="https://github.com/basel-shehabi/EEG-Synthetic-Data-Using-GANs/blob/master/pics/fig3.PNG">
</p>

Standard GANs without any modifications are prone to errors and instability, more specifically what is known as mode collapse. Mode collapse is defined as the discriminator only learning to recognize a few narrow modes of input distribution as real, causing the generator to produce a limited amount of differentiable outputs. One proposition to counteract model instability and mode collapse is using a Wasserstein GAN (WGAN) that was first introduced by Arjovsky et al. The end goal of WGANs is to minimise the difference between real data distribution and fake data distribution, preventing mode collapse, and has been successfully used to create noiseless EEG data. There are also other issues such as vanishing/exploding gradients that a WGAN can take care of using what is known as a gradient penalty. More wil be touched on this in the methodology section. Other researchers might use a pure Deep Convolutional GAN (DCGAN) in order to pick up on signal features using convolutional layers with different kernel sizes. Based on this, three kinds of GAN were implemented and investigated: *a standard unmodified GAN, a DCGAN and a WGAN.*

In this project, the focus was on what is known as a Endogenous BCI system -- BCI system that uses ERD signals, more specifically Sensorimotor Rhythms (SMR), as a method of input. SMR signals are signals that either grow or decrease in power/amplitude typically around the Mu and Beta oscillatory rhythms (8-13Hz and 13-30 Hz) over the motor cortex regions of the brain. This decrease in power is noted when a subject or a person is imagining/thinking about movement, hence, the BCI system relies on **Motor Imagery (MI)** to use. MI-based BCI systems present the user or subject with a cue such as an arrow that indicates which limb to move (e.g. left arrow for left arm) over a specified period of time (typically 4-6 seconds). This constitutes what is known as a single trial. Usually multiple trials around (300-400) are required in order to gather enough signal data for the system to use. 

# Methodology

As mentioned previously, the focus of this project was on creating synthetic/artificial data for use in MI-based BCI systems. 13 Subjects were each individually invited to sit in front of a computer screen and perform the imagery task in their head over several trials. One trial had consisted of roughly 5.5-6.5 seconds and began with an audible beep heard at t = 0 seconds, followed by an arrow on the screen pointing towards the left, right or up, indicating which arm to imagine moving (up indicates both arms or bimanual). The arrow had stayed on the screen for a period of four seconds with a blank screen shown for a random duration of 1.5-2.5 seconds. The trial can be demonstrated in figure 4. 

<p align="center">
  <img src="https://github.com/basel-shehabi/EEG-Synthetic-Data-Using-GANs/blob/master/pics/fig4.PNG">
</p>

For each subject, 315 trials were collected, with 105 trials pertaining to each task (e.g. 105 trials for left arm movements). This was split into 7 runs of 45 trials in order to minimise subject fatigue. Data was collected from 64 EEG channels and sampled at 256Hz. 

Following that, the collected data was stored on a computer and explored using the MNE Python Module/Package which allows easy exploration of EEG data. The data was pre-processed and split into what is known as 'Epochs' or trials where noisy/unsuitable trials were dropped and a time frame of -2 to 5 seconds per trial was selected. 

The overall trial data was then filtered to match the frequency bands of interest: 8-30 Hz (corresponding the the frequency bands of the Sensorimotor Rhythm signals), and split equally per task where an equal number of trials was present per subject for left, right and bimanual imagery. Moreover, the data was also cropped between 0-4 seconds to match the time in which the cue was present. Data had then undergone Continuous Wavelet Transform (CWT) in order to pick up on the transient signal behaviour as well as serving as a good method for feature extraction. Finally, the data was normalised around a value of -1 to 1 such that the neural networks can learn at a much faster rate. NOTE that only data from one channel was picked and this is due to limitations faced by the machine which ran the networks. 

Once the processing part was done, scikit learn's `train_test_split()` method had split the trial data per subject into a training set and a testing dataset (that was later used as validation) into a 75% training and 25% testing and saved into .npz files containing two arrays: one having the trial data per task and subject in the form of (Trial sub-array x Frequency Sample sub-array x Time Sample sub-array x Channel sub-array) and the other having the task label associated. For instance, a file called Subject1Train1.npz corresponds to training data for subject 1, task 1 (left imagined movements). Left movements are one, right movements two and bimanual are three. 

The overall implementation can be found in the `Load_and_Process.py` script. The script implements the LoadData class with three methods, one to return the filtered, normalised and transformed data using `get_normalized_cwt_data()` which accepts the channel numbers based off of the class constructor and defaults to channels 7, 9 and 11 if nothing was passed. The two other class methods `get_epoch_data()` and `get_crop_filtered_data_from_fif()` return the overall trial data for one subject that is uncropped, non-split and unfiltered having only gone pre-processing from MNE, while the latter only splits, crops and filters without undergoing CWT. This is done for exploratory purposes where initially the standard GAN was only fed data returned from `get_epoch_data()` which was in the form of Trials x Time Samples x Channels. Figure 5 shows data figures in the form of Time Series, Image Plot and Spectrogram from select trials and channels, and can be obtained using the same load script.

<p align="center">
  <img src="https://github.com/basel-shehabi/EEG-Synthetic-Data-Using-GANs/blob/master/pics/fig5.PNG">
</p>

<p align="center">
  <img src="https://github.com/basel-shehabi/EEG-Synthetic-Data-Using-GANs/blob/master/pics/fig6.png">
</p>

<p align="center">
  <img src="https://github.com/basel-shehabi/EEG-Synthetic-Data-Using-GANs/blob/master/pics/fig7.png">
</p>

### Network Architecture and Parameters




# Results

# Discussion & Conclusion

# Future Work

# Usage/Installation


# Acknowledgements
My Supervisors: Dr. Aleksandra Vuckovic and Professor Iain Thayne

Fellow colleague and student: Ciaran McGeady

Code snippets and tutorials:

* [Jeff Heaton](https://github.com/jeffheaton/t81_558_deep_learning)
* [Erik Linder-Noren](https://github.com/eriklindernoren/Keras-GAN)
* [Drew Szurko](https://github.com/drewszurko/tensorflow-WGAN-GP)
* [Krish Kabra](https://github.com/krishk97/ECE-C247-EEG-GAN)

