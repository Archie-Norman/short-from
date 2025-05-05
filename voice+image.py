import os
import subprocess

output_folder = r"F:\TikToks\surely"
audio_file = r"F:\TikToks\voiceover_1.wav"
video_file = r"F:\TikToks\slideshow.mp4"

# Generate a text file listing all images in the correct order
image_list_file = os.path.join(output_folder, "images.txt")

# Get all PNG images, sorted
images = sorted([f for f in os.listdir(output_folder) if f.endswith(".png")])

# Write the list of images to a file
with open(image_list_file, "w") as f:
    for img in images:
        f.write(f"file '{os.path.join(output_folder, img)}'\n")

# FFmpeg command to create video from images
ffmpeg_cmd = [
    "ffmpeg",
    "-f", "concat",
    "-safe", "0",
    "-i", image_list_file,
    "-i", audio_file,
    "-c:v", "libx264",
    "-r", "30",
    "-pix_fmt", "yuv420p",
    "-shortest",
    video_file
]

# Run FFmpeg
subprocess.run(ffmpeg_cmd)

print(f"Slideshow video created: {video_file}")
