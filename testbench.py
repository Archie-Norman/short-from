from llama_cpp import Llama
import re
from diffusers import StableDiffusionPipeline
import torch
import numpy as np
from PIL import Image
import os
import torchaudio
import torch
from bark import generate_audio, preload_models
from moviepy.editor import ImageSequenceClip, AudioFileClip

model_path = "c:/Users/archi/Desktop/TikToks/mistral-7b-instruct-v0.1.Q8_0.gguf"

# Load the model
llm = Llama(model_path=model_path)

# Define a statement that is wrong
trend = "key word"

# Create a prompt asking the model to explain why the statement is wrong
prompt = f"Write a 30-second random speech about {trend}. no need for an introduction just get straight into it"

# Generate a response and store the output
response = llm(prompt, max_tokens=2048, temperature=0.7, top_p=0.9, top_k=50, repeat_penalty=1.1)

# Extract the text output
speech = response["choices"][0]["text"].strip()


# Use regex to split the output into sentences
output_list = re.findall(r"([^.?!]*[.?!])", speech)


print("Generated the speech")


import re

def clean_speech(text):
    # Regular expression to match the first word and everything after it
    cleaned_text = re.sub(r'^[^a-zA-Z]*(\w.*)', r'\1', text)
    
    # Remove commas in lists, but not in other cases (like 'I'm')
    cleaned_text = re.sub(r'(\w), (\w)', r'\1 \2', cleaned_text)
    
    return cleaned_text

cleaned_speech = clean_speech(speech)
print(cleaned_speech)
##################################################################################################################################################################################################################
# create the output folder

# Ensure there are sentences to avoid IndexError
if len(output_list) > 1:
    folder_name = output_list[1]  # Get the second sentence
else:
    folder_name = output_list[0]  # Fall back to the first sentence if no second sentence exists

# Sanitize the folder name to make it filesystem-friendly (remove special characters and spaces)
sanitized_folder_name = re.sub(r'[^\w\s-]', '', folder_name).strip()

# Ensure the folder name isn't empty after sanitization
if not sanitized_folder_name:
    sanitized_folder_name = "default_folder"

# Set the output folder path
output_folder = f"c:/Users/archi/Desktop/TikToks/outputs/{sanitized_folder_name}"

# Create the folder
os.makedirs(output_folder, exist_ok=True)

# Print the output folder path
print(f"Output folder created at: {output_folder}")


##################################################################################################################################################################################################################
num = 0
outputs = []

# Replace with your desired trend

while num < 5:  # Repeat the process until 5 prompts are collected
    # Create the prompt to generate a random image related to the trend
    prompt = f"give me a detailed prompt to generate a random image related to {trend}. no text in the images. be extremely detailed and descriptive. no more than 77 tokens"
    
    # Generate the response from the model
    response = llm(prompt, max_tokens=2048, temperature=0.7, top_p=0.9, top_k=50, repeat_penalty=1.1)

    # Extract the response text
    output = response["choices"][0]["text"].strip()
    
    # Add the output to the outputs list
    outputs.append(output)

    # Increment the number of iterations
    num += 1

print("Generated the prompts for the images")


##################################################################################################################################################################################################################


# Path to the converted model directory
model_path = "F:/TikToks/stable-diffusion-v1-4"  # Ensure this points to the correct model folder

# Load the Stable Diffusion pipeline
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32)
pipe.to("cpu")  # Use CPU since you're on an AMD RX580

# Desired resolution for the output images
image_width, image_height = 768, 768  # Smaller image resolution

# Use PLMS sampler for faster image generation
pipe.scheduler.set_timesteps(50)  # Reduce denoising steps to 50 (for faster generation)

# Loop through each prompt in the list and generate an image
for idx, prompt in enumerate(outputs):
    # Generate the image
    image = pipe(prompt, num_inference_steps=50).images[0]

    # Create the folder based on the sanitized output
    sanitized_folder_name = re.sub(r'[^\w\s-]', '', output_list[1]).strip()
    sanitized_folder_name = sanitized_folder_name if sanitized_folder_name else "default_folder"
    output_folder = f"c:/Users/archi/Desktop/TikToks/outputs/{sanitized_folder_name}"

    # Create the output folder for each set of images
    os.makedirs(output_folder, exist_ok=True)  # Create the folder if it doesn't exist

    # Save the image in the output folder
    image_path = os.path.join(output_folder, f"generated_image_{idx + 1}.png")
    image.save(image_path)

    print(f"Image saved at {image_path}")

    # Optionally display the image
    image.show()

print("Generated the images")


##################################################################################################################################################################################################################

from TTS.api import TTS

# Choose a model (replace with the one you want)
model_name = "tts_models/en/ljspeech/tacotron2-DDC"

# Initialize the TTS model (downloads automatically)
tts = TTS(model_name)

# Ensure the output folder exists (we use the same folder as for the images)
sanitized_folder_name = re.sub(r'[^\w\s-]', '', output_list[1]).strip()
sanitized_folder_name = sanitized_folder_name if sanitized_folder_name else "default_folder"
output_folder = f"c:/Users/archi/Desktop/TikToks/outputs/{sanitized_folder_name}"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Save the audio file in the output folder
audio_path = os.path.join(output_folder, "voiceover_1.wav")

# Generate speech
tts.tts_to_file(text= cleaned_speech, file_path=audio_path)


print(f"Generated the voiceover at {audio_path}")



##################################################################################################################################################################################################################

# Load all image files from the output folder
images = sorted([os.path.join(output_folder, f) for f in os.listdir(output_folder) if f.endswith(('jpg', 'png'))])

# Load the audio from the output folder
audio_path = os.path.join(output_folder, "voiceover_1.wav")
audio = AudioFileClip(audio_path)
audio_duration = audio.duration  # Get duration of the audio

# Calculate duration per image
num_images = len(images)
image_duration = audio_duration / num_images

# Create the slideshow
clip = ImageSequenceClip(images, durations=[image_duration] * num_images)
clip = clip.set_audio(audio)  # Add audio to the slideshow

# Export the video
output_video_path = os.path.join(output_folder, "slideshow.mp4")
clip.write_videofile(output_video_path, fps=24, codec="libx264")

print(f"Generated the slideshow with voice over at {output_video_path}")


##################################################################################################################################################################################################################