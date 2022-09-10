import os
from tabnanny import filename_only 
import numpy as np
from PIL import Image


# resize images to size 128, 
#image channels to 3,
# and save them to a new folder

IMAGE_SIZE = 128
IMAGE_CHANNELS = 3
IMAGE_DIR = 'data/impressionism'


images_path = IMAGE_DIR
images_channels = IMAGE_CHANNELS
images_size = IMAGE_SIZE

def resize_images(image_size, image_channels, images_path):

    training_data = []

    #iterate over the images inside directory and resize using pillow resize mething

    # os.listdir(images_path)

    for folder in next(os.walk(images_path))[1]:
        for filename in os.listdir(os.path.join(images_path,folder)):
            print("..... resizing.....")
            print(filename)
            path = os.path.join(images_path, folder, filename)
            print(f"{path} is the path of file")
            image = Image.open(path)
            image = image.resize((image_size, image_size),
                                 Image.ANTIALIAS)
            # training_data.append(np.asarray(image))
            np.append(training_data, np.asarray(image))
            training_data = np.reshape(training_data, 
                                        (-1, image_size, image_size, image_channels))
            training_data = training_data / 127.5 - 1
            
            print('saving file.......')
            np.save(
            'impressionism.npy', training_data)
            
            
       

if __name__=='__main__':
    resize_images(images_size, images_channels, images_path)
