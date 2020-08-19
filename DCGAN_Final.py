import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
from keras.models import Sequential
from keras import layers, Input

# Load the data and store it appropriately

train_dataset_task1 = np.load('Subject1Train1.npz')

print(train_dataset_task1.get('y'))

train_data = train_dataset_task1['x']
train_labels = train_dataset_task1['y']

class DCGAN:

    # Hyper-parameter and model setup

    def __init__(self, channels=1, batchsize=50):

        # Dataset features:
        self.channels = channels
        self.freq_sample = 25
        self.time_sample = 342
        self.eeg_shape = (self.freq_sample, self.time_sample, self.channels)

        # Model specific parameters (Noise generation, Dropout for overfitting reduction, etc...):
        self.noise = 100
        self.dropout = 0.3 # 0.2 original
        self.momentum = 0.8 # 0.8 original
        self.batchsize = batchsize

        # Choosing Adam optimiser for both generator and discriminator to feed in to the model:
        self.optimiser = Adam(0.0002, 0.2) # Values from the EEG GAN paper found to be most optimal

        # Build both the Generator and Discriminator:
        self.generator = self.make_generator()
        self.discriminator = self.make_discriminator()

        # Will be used to store useful information about the training session such as loss and accuracy
        self.info = {}

    def make_generator(self):

        '''
        Creates a generator model that takes in randomly generated noise, then uses
        3 upsampling layers to return an image that is fed into the discriminator
        which then distinguishes whether or not it is a real or fake one. Weights are adjusted
        accordingly such that it can eventually generate a real signal.
        :return:
        '''

        model = Sequential()

        model.add(layers.Dense(4 * 41 * 256, use_bias=False, input_shape=(self.noise,)))
        model.add(layers.BatchNormalization(momentum=self.momentum))
        model.add(layers.LeakyReLU())

        model.add(layers.Reshape((4, 41, 256)))
        assert model.output_shape == (None, 4, 41, 256)  # Assertions help us ensure our model is working correctly

        model.add(layers.Conv2DTranspose(128, (5, 4), strides=(2, 2), padding='valid', use_bias=False))
        model.add(layers.BatchNormalization(momentum=self.momentum))
        model.add(layers.LeakyReLU())
        assert model.output_shape == (None, 11, 84, 128)

        model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='valid', use_bias=False))
        model.add(layers.BatchNormalization(momentum=self.momentum))
        model.add(layers.LeakyReLU())
        assert model.output_shape == (None, 25, 171, 64)

        model.add(layers.Conv2DTranspose(self.channels, (5, 5), strides=(1, 2), padding='same', use_bias=False,
                                         activation='tanh')) # Using tanh for output also based on the EEG paper
        assert model.output_shape == (None, 25, 342, self.channels)

        # Prints a small summary of the network
        model.summary()

        noise = Input(shape=(self.noise,))
        img = model(noise)

        return model

    def make_discriminator(self):

        '''
        Creates a discriminator model that distingushes the fed images from generator,
        and also is trained using a training loop (see below). The Discriminator is a simple
        2 layer CNN that returns either a 'True' or 'False'. Values are then adjusted accordingly
        per epoch to update weights and biases such that it produces the right output (i.e. it can
        discriminate fake from real).
        :return:
        '''

        model = Sequential()
        model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                input_shape=[25, 342, self.channels]))
        model.add(layers.BatchNormalization(momentum=self.momentum))
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2D(128, (5, 5), strides=(1, 2), padding='same'))
        model.add(layers.BatchNormalization(momentum=self.momentum))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(self.dropout))
        print(model.output_shape)

        model.add(layers.Flatten())
        model.add(layers.Dense(1))
        assert model.output_shape == (None, 1)

        img = Input(shape=self.eeg_shape)
        validity = model(img)

        return model

    def make_fakedata(self, N=100):

        '''
        Generates the fake data after training
        :return:
        '''

        noise = np.random.normal(0, 1, (N, self.noise))
        gen_imgs = self.generator.predict(noise)
        return gen_imgs, noise


    def discriminator_loss(self, real_output, fake_output):

        '''
        Defines the loss function for the descriminator.
        Uses cross entropy a.k.a (log-loss) helper function from
        Keras 'BinaryCrossEntropy'. Returns total loss.
        '''
        cross_entropy = BinaryCrossentropy(from_logits=True)
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output):

        '''
        Like the above but this time for generator...
        :return:
        '''

        cross_entropy = BinaryCrossentropy(from_logits=True)
        return cross_entropy(tf.ones_like(fake_output), fake_output)#

    @tf.function
    def train_step(self, images):

        '''
        This training step function that follows from the official TensorFlow documentation.
        It is in the form of tf.function which allows it to be compiled, rather than
        compiling each of the models alone everytime.
        :return:
        '''

        disc_loss, gen_loss = 0,0 # Store disc and gen loss values

        # GradientTape allows us to do automatic differentiation handled by TensorFlow
        # Useful when doing back propagation obviously. It also watches all the differentiable
        # Variables

        noise = tf.random.normal([self.batchsize, self.noise])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)

            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        grad_gen = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        grad_disc = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.optimiser.apply_gradients(zip(grad_gen, self.generator.trainable_variables))
        self.optimiser.apply_gradients(zip(grad_disc, self.discriminator.trainable_variables))

        return disc_loss, gen_loss

    def train(self, dataset, epochs):

        '''
        The training function that has a loop which trains the model on
        every epoch/iteration.
        '''

        # Allows us to 'unpack' our dataset using .from_tensor_slices, shuffling it
        # and also batching it.

        info = self.info
        info['loss'] = {}
        gen_loss, disc_loss = [], []

        data = tf.data.Dataset.from_tensor_slices(dataset.astype('float32'))\
                    .shuffle(dataset.shape[0]).batch(self.batchsize)

        for epoch in range(epochs):

            for image_batch in data:
                disc_loss_batch, gen_loss_batch = self.train_step(image_batch)

                gen_loss.append(gen_loss_batch)
                disc_loss.append(disc_loss_batch)

            gen_loss_tot = sum(gen_loss) / len(gen_loss)
            d_loss_tot = sum(disc_loss) / len(disc_loss)

            if epoch % 100 == 0:
                print("epoch: {}, generator loss: {}, discriminator loss: {} accuracy: {}".format
                     (epoch, gen_loss_tot, d_loss_tot, 100*d_loss_tot))

                '''
                # fake image example
                generated_image, _ = self.make_fakedata(N=1)
                # real image example
                trial_ind, eeg = 30, 0
                real_image = np.expand_dims(dataset[trial_ind], axis=0)

                # visualize fake and real data examples
                plt.figure()
                plt.subplot(121)
                plt.imshow(generated_image[0, :, :, eeg], aspect='auto')
                plt.colorbar()
                plt.title('Artificial Data')
                plt.subplot(122)
                plt.imshow(real_image[0, :, :, eeg], aspect='auto')
                plt.title('Real Data')
                plt.colorbar()
                plt.subplots_adjust(hspace=0.5)
                plt.show()
                '''

        plt.figure()
        plt.plot(gen_loss, 'r')
        plt.plot(disc_loss, 'b')
        plt.title('Loss history')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(['Generator', 'Discriminator'])
        plt.show()


something = DCGAN()
x = something.train(train_data, epochs=2000)


