import os 
import numpy as np

from PIL import Image
from runs.nn import generator, discriminator, GAN
from utils.saveDisplayFiles import save_imgs


#Constants
LATENT_DIM = 100


# Paths 
path = "data/resized_data/"
save_model_path = "models/"


def train(path, epochs, batch_size=32, save_interval=200):

  array = []


  for dir in os.listdir(path):
            # print(dir)
    image = Image.open(path + dir)
    data = np.asarray(image)
    array.append(data)

  X_train = np.array(array)
  print(X_train.shape)

  # print(X_train.shape)
  #Rescale data between -1 and 1
  X_train = X_train / 127.5 -1.
  bat_per_epo = int(X_train.shape[0] / batch_size)
  # X_train = np.expand_dims(X_train, axis=3)
  print(X_train.shape)

  #Create our Y for our Neural Networks
  valid = np.ones((batch_size, 1))
  fakes = np.zeros((batch_size, 1))

  for epoch in range(epochs):
    for j in range(bat_per_epo):
      #Get Random Batch
      idx = np.random.randint(0, X_train.shape[0], batch_size)
      imgs = X_train[idx]

      #Generate Fake Images
      noise = np.random.normal(0, 1, (batch_size, LATENT_DIM))
      gen_imgs = generator.predict(noise)

      #Train discriminator
      d_loss_real = discriminator.train_on_batch(imgs, valid)
      d_loss_fake = discriminator.train_on_batch(gen_imgs, fakes)
      d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

      noise = np.random.normal(0, 1, (batch_size, LATENT_DIM))
      
      #inverse y label
      g_loss = GAN.train_on_batch(noise, valid)

      print("******* %d %d [D loss: %f, acc: %.2f%%] [G loss: %f]" % (epoch,j, d_loss[0], 100* d_loss[1], g_loss))

      # if(epoch % save_interval) == 0:
      save_imgs(epoch)



def save_model_weights(save_model_path):
    generator.save_weights(save_model_path + "generator.h5")
    discriminator.save_weights(save_model_path + "discriminator.h5")
    GAN.save_weights(save_model_path + "GAN.h5")