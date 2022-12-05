# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

from ast import parse
import os
from pyexpat import model
import re
from typing import List, Optional, Tuple, Union
import numpy as np
import PIL.Image
import torch
from networks_fastgan import MyGenerator
import random
#----------------------------------------------------------------------------

def parse_range(s: Union[str, List]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.
    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

def parse_vec2(s: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
    '''Parse a floating point 2-vector of syntax 'a,b'.
    Example:
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    parts = s.split(',')
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise ValueError(f'cannot parse 2-vector {s}')

#----------------------------------------------------------------------------

def make_transform(translate: Tuple[float,float], angle: float):
    m = np.eye(3)
    s = np.sin(angle/360.0*np.pi*2)
    c = np.cos(angle/360.0*np.pi*2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m

#----------------------------------------------------------------------------

def generate_images(
    model_path,
    number_of_images,
    seeds = "10-12",
    truncation_psi = 1.0,
    noise_mode = "const",
    outdir = "out",
    translate = "0,0",
    rotate = 0
):
    model_owner = "huggan"

    model_path_dict = {
        'Impressionism' : 'projected_gan_impressionism',
        'Cubism' : 'projected_gan_cubism', 
        'Abstract Expressionism' : 'projected_gan_abstract_expressionism', 
        'Pop Art' : 'projected_gan_popart',
        'Minimalism' : 'projected_gan_minimalism',    
    }
    
    model_path = model_owner + "/" + model_path_dict[model_path]
    print(model_path)
    print(seeds)
    seeds=random.randint(1,230)
    seeds =f"{seeds}-{seeds+number_of_images-1}"
    seeds = parse_range(seeds)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    G = MyGenerator.from_pretrained(model_path)
    os.makedirs(outdir, exist_ok=True)
    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)
    """
    if G.c_dim != 0:
        if class_idx is None:
            raise click.ClickException('Must specify class label with --class when using a conditional network')
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print ('warn: --class=lbl ignored when running on an unconditional network')
    """
    # Generate images.

    #imgs_row = np.array()
    #imgs_complete = np.array()
    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device).float()
        # Construct an inverse rotation/translation matrix and pass to the generator.  The
        # generator expects this matrix as an inverse to avoid potentially failing numerical
        # operations in the network.
        if hasattr(G.synthesis, 'input'):
            m = make_transform(translate, rotate)
            m = np.linalg.inv(m)
            G.synthesis.input.transform.copy_(torch.from_numpy(m))
        
        img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        print(seed_idx)
        #first image
        if seed_idx == 0:
            imgs_row = img[0].cpu().numpy()
        else:
            imgs_row = np.hstack((imgs_row, img[0].cpu().numpy()))
        # img = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB')
        # PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/seed{seed:04d}.png')
    #napravi vsplit i toe to ka
    imgs_complete = np.vstack(np.hsplit(imgs_row, 4))
    #cv2.imshow("lalaxd", imgs_complete)
    #cv2.waitKey()
    return PIL.Image.fromarray(imgs_complete, 'RGB')


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------