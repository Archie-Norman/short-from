from llama_cpp import Llama
import re
from diffusers import StableDiffusionPipeline
import torch
import numpy as np
from PIL import Image
import os


model_path = "F:/TikToks/mistral-7b-instruct-v0.1.Q8_0.gguf"

# Load the model
llm = Llama(model_path=model_path)

# Define a statement that is wrong

num = 0
outputs = []

# Replace with your desired trend
trend = "Tkey-wordrump"  # or any other trend you'd like

while num < 5:  # Repeat the process until 5 prompts are collected
    # Create the prompt to generate a random image related to the trend
    prompt = f"give me a detailed prompt to generate a random image related to {trend}. no text in the images. be extremely detailed and descriptive"
    
    # Generate the response from the model
    response = llm(prompt, max_tokens=2048, temperature=0.7, top_p=0.9, top_k=50, repeat_penalty=1.1)

    # Extract the response text
    output = response["choices"][0]["text"].strip()
    
    # Add the output to the outputs list
    outputs.append(output)

    # Increment the number of iterations
    num += 1

print(outputs)
print("4")