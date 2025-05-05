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

#################################################################################################################
import re

def clean_speech(text):
    # Regular expression to match the first word and everything after it
    cleaned_text = re.sub(r'^[^a-zA-Z]*(\w.*)', r'\1', text)
    
    # Remove commas in lists, but not in other cases (like 'I'm')
    cleaned_text = re.sub(r'(\w), (\w)', r'\1 \2', cleaned_text)
    
    return cleaned_text

cleaned_speech = clean_speech(speech)
print(cleaned_speech)

#################################################################################################################
print(speech)

from TTS.api import TTS

# Choose a model (replace with the one you want)
model_name = "tts_models/en/ljspeech/tacotron2-DDC"

# Initialize the TTS model (downloads automatically)
tts = TTS(model_name)

# Generate speech
tts.tts_to_file(text= cleaned_speech, file_path="output.wav")
