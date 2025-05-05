from diffusers import StableDiffusionPipeline
import torch
import numpy as np
from PIL import Image

# Path to the converted model directory
model_path = "C:/Users/archi/Desktop/TikToks/stable-diffusion-v1-4"  # Ensure this points to the correct model folder

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32)
pipe.to("cpu")

pipe.to("cpu")  # Use CPU since you're on an AMD RX580

# Define the text prompt
prompt = "a man getting out of a car"

# Generate the image
image = pipe(prompt).images[0]

# Convert PIL image to numpy array and display using PIL
image.show()
