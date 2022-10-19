import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Reshape, Flatten, Conv2D, Dense, Conv2DTranspose, Dropout, LeakyReLU
from tensorflow.keras.optimizers import Adam


# constants 
LATENT_DIM = 100
IMG_SHAPE = (64,64,3)

#optimizer
adam = Adam(lr=0.0002, beta_1=0.5)

# generator network 

def build_generator():
    model = Sequential()
    model.add(Dense(256 * 8* 8, input_dim=LATENT_DIM))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((8,8,256)))

    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(3, (3,3), activation='tanh', padding='same'))
  
    model.summary()

    return model

# discriminator network

def build_discriminator():
    model = Sequential()
    model.add(Conv2D(64, (3,3), padding='same', input_shape=IMG_SHAPE))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(128, (3,3), padding='same', ))
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Conv2D(128, (3,3), padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(256, (3,3), padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))

    model.summary()
    return model


def generator(): 

    generator = build_generator()

    return generator

def discriminator():

    discriminator = build_discriminator()
    discriminator.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

    return discriminator

def GAN():

    discriminator = build_discriminator()
    discriminator.trainable = False

    generator = build_generator()

    gan = Sequential()
    gan.add(generator)
    gan.add(discriminator)

    gan.compile(loss='binary_crossentropy', optimizer=adam)

    return gan