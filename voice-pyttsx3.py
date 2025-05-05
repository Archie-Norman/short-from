import pyttsx3

# Initialize the pyttsx3 engine
engine = pyttsx3.init()

# Set properties for a natural voice (you can adjust these to get the right sound)
engine.setProperty('rate', 150)  # Speed of speech (default is 200)
engine.setProperty('volume', 1)  # Volume level (1.0 is max, 0.0 is min)

# Optional: Change the voice to a more natural-sounding one
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)  # Change index to use a different voice (1 is typically a more natural female voice)

# List of image prompts (replace this with your generated prompts)
image_prompts = [
    "The current administration has been plagued by controversy and scandal from day one, with the president's divisive rhetoric and actions causing tension both at home and abroad. The ongoing investigations into his businesses and personal relationships only add to the uncertainty surrounding his leadership. Despite these challenges, many Americans continue to support him, hoping for change and progress in their lives. However, it is important to remember that the president's words and actions have real consequences, and we must hold him accountable for his actions."
]

# Iterate over the prompts and generate voiceover for each
for idx, prompt in enumerate(image_prompts, start=1):
    print(f"Generating voiceover for prompt {idx}: {prompt}")
    engine.save_to_file(prompt, f"voiceover_{idx}.mp3")

# Run the engine (this will generate and save the voiceovers)
engine.runAndWait()

print("Voiceovers saved as MP3 files.")
