import os
import re
import torch
import numpy as np
import torchaudio
import subprocess
from PIL import Image
from llama_cpp import Llama
from diffusers import StableDiffusionPipeline
from bark import generate_audio, preload_models

# Paths
llm_model_path = "F:/TikToks/mistral-7b-instruct-v0.1.Q8_0.gguf"
sd_model_path = "F:/TikToks/stable-diffusion-v1-4"
output_folder = "F:/TikToks/output"
audio_file = os.path.join(output_folder, "voiceover.wav")
video_file = os.path.join(output_folder, "slideshow.mp4")

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Load LLaMA model
llm = Llama(model_path=llm_model_path)

# Define trend
trend = "key-word"

# Generate speech text
prompt = f"Write a 30-second random speech about {trend}. No need for an introduction, just get straight into it."
response = llm(prompt, max_tokens=2048, temperature=0.7, top_p=0.9, top_k=50, repeat_penalty=1.1)
speech_text = response["choices"][0]["text"].strip()

# Save text output
print("Generated Speech Text:\n", speech_text)

# Generate descriptive image prompts
image_prompts = []
for i in range(5):  # Generate 5 image prompts
    prompt = f"Give me a detailed prompt to generate a random image related to {trend}. No text in the images. Be extremely detailed and descriptive."
    response = llm(prompt, max_tokens=2048, temperature=0.7, top_p=0.9, top_k=50, repeat_penalty=1.1)
    image_prompts.append(response["choices"][0]["text"].strip())

print("Generated Image Prompts:\n", image_prompts)

# Load Stable Diffusion pipeline
pipe = StableDiffusionPipeline.from_pretrained(sd_model_path, torch_dtype=torch.float32)
pipe.to("cpu")  # Use CPU for AMD RX580

# Generate images
image_paths = []
for idx, prompt in enumerate(image_prompts, start=1):
    print(f"Generating image {idx}...")
    image = pipe(prompt, num_inference_steps=50).images[0]
    image_path = os.path.join(output_folder, f"image_{idx}.png")
    image.save(image_path)
    image_paths.append(image_path)

print("All images saved successfully.")

# Load Bark AI model
preload_models()

# Generate voiceover
audio_array = generate_audio(speech_text)
audio_tensor = torch.from_numpy(audio_array).unsqueeze(0)  # Add batch dimension

# Set torchaudio backend and save
torchaudio.set_audio_backend("sox_io")
torchaudio.save(audio_file, audio_tensor, 24000)

print("Voiceover saved successfully.")

# Generate slideshow using ffmpeg
image_pattern = os.path.join(output_folder, "image_%d.png")
ffmpeg_cmd = [
    "ffmpeg",
    "-framerate", "1",  # 1 second per image
    "-i", image_pattern,
    "-i", audio_file,
    "-c:v", "libx264",
    "-r", "30",
    "-pix_fmt", "yuv420p",
    "-shortest",
    video_file
]

# Run ffmpeg
subprocess.run(ffmpeg_cmd)

print(f"Slideshow video created: {video_file}")
