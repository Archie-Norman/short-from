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
trend = "key word"

# Create a prompt asking the model to explain why the statement is wrong
prompt = f"Write a 30-second random speech about {trend}. no need for an introduction just get straight into it"

# Generate a response and store the output
response = llm(prompt, max_tokens=2048, temperature=0.7, top_p=0.9, top_k=50, repeat_penalty=1.1)

# Extract the text output
output = response["choices"][0]["text"].strip()


# Use regex to split the output into sentences
output_list = re.findall(r"([^.?!]*[.?!])", output)

print("1")
print(output)
print("2")
print(output_list)
print("3")

