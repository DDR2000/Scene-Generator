import torch
import os
from diffusers import StableDiffusionXLPipeline
import sys


def read_prompts(filename):
    try:
        with open(filename, 'r') as file:
            prompt_text = file.read()
            print("Prompt read successfully:")
            print(prompt_text)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


create_directory("presentation")
create_directory("presentation/sdxl")
create_directory("presentation/photo_sdxl")
# Load Stable Diffusion XL
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16)

pipe = pipe.to("cuda")

if len(sys.argv) != 2:
    print("Usage: python script.py prompts.txt")
else:
    prompt_file = sys.argv[1]
    read_prompts(prompt_file)

    # Read prompts from a file
    with open(prompt_file, "r") as file:
        prompts = file.read().splitlines()


# Process each set of prompts
for i, prompt in enumerate(prompts):
    # Generate an image using Stable Diffusion XL
    image = pipe(prompt).images[0]
    filename = f"presentation/sdxl/sdxl_image_{i}.png"
    image.save(filename)

    # Prompt addition
    photo_prompt = "Normal Angle, Mirrorless, 55mm, f/1.8, shutter 1/30, iso 10000"
    prompt = prompt + " " + photo_prompt

    # Generate an image using Stable Diffusion XL
    image = pipe(prompt).images[0]

    filename = f"presentation/photo_sdxl/sdxl_photo_image_{i}.png"
    image.save(filename)

