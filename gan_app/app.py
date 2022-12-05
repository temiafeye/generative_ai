import gradio as gr
from huggingface_hub import PyTorchModelHubMixin
import torch
import matplotlib.pyplot as plt
import torchvision
from networks_fastgan import MyGenerator
import click
import PIL
from image_generator import generate_images

def image_generation(model, number_of_images=8):
    img = generate_images(model, number_of_images=number_of_images)
    return img

if __name__ == "__main__":
    description = "A Simple Digital Art Gallery to showcase the work of the ArtGAN model across diffent classical art styles."
    inputs = [gr.inputs.Radio([ "Impressionism", "Abstract Expressionism", "Cubism", "Pop Art"]), gr.Slider(8, 64, value=8, step=4),]
    outputs = gr.outputs.Image(label="Generated Image", type="pil")


    title = "Digital Art Gallery With ArtGAN"
    article = "<p style='text-align: center'><a href='https://github.com/temiafeye/generative_ai'>Computer Generated Art with GANs + Paper By Temi Afeye, 2022</a></p>"



    gr.Interface(image_generation, inputs, outputs, title=title, article = article, description = description, allow_flagging="never",
    analytics_enabled=False).launch(debug=True)

    app, local_url, share_url = iface.launch()