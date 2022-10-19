import matplotlib.pyplot as plt
import numpy as np
import glob 
import imageio
import PIL

from runs import nn


save_name = 0.0000000

LATENT_DIM = 100

def save_imgs(epoch): 

    r, c = 4,4 
    noise = np.random.normal(0, 1, (r * c, LATENT_DIM))
    gen_imgs = nn.generator.predict(noise)

    global save_name

    save_ame += 0.000000001

    # Rescale images 0 - 1
    gen_imgs = (gen_imgs + 1) / 2.0

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt, :,:,:])
            axs[i,j].axis('off')
            cnt += 1

    fig.savefig("currentgeneration.png")
    fig.savefig("generated_images/%.8f.png" % save_name)
    plt.close()
