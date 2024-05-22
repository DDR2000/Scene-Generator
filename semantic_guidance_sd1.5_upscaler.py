import torch
import os
from diffusers import SemanticStableDiffusionPipeline
from diffusers import StableDiffusionUpscalePipeline
import sys


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


create_directory("presentation/semantic_sd15")
create_directory("presentation/semantic_sd15/upscaled")

# load model and scheduler
model_id = "stabilityai/stable-diffusion-x4-upscaler"
pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipeline = pipeline.to("cuda")

semantic_pipe = SemanticStableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
    )
semantic_pipe = semantic_pipe.to("cuda")


def read_prompts(filename):
    try:
        with open(filename, 'r') as file:
            prompt_text = file.read()
            print("Prompt read successfully:")
            print(prompt_text)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")


def semantic_guidance(small_prompt, editing_prompt):
    out = semantic_pipe(
        prompt=small_prompt,
        num_images_per_prompt=1,
        guidance_scale=7,
        editing_prompt=editing_prompt,
        reverse_editing_direction=[
            False,
            False,
            False,
            False,
        ],  # Direction of guidance i.e. increase all concepts
        edit_warmup_steps=[10, 10, 10, 10],  # Warmup period for each concept
        edit_guidance_scale=[4, 5, 5, 5.4],  # Guidance scale for each concept
        edit_threshold=[
            0.99,
            0.975,
            0.925,
            0.96,
        ],
        # Threshold for each concept. Threshold equals the percentile of the latent space that will be discarded.
        # I.e. threshold=0.99 uses 1% of the latent dimensions
        edit_momentum_scale=0.3,  # Momentum scale that will be added to the latent guidance
        edit_mom_beta=0.6,  # Momentum beta
        edit_weights=[1, 1, 1, 1, 1],  # Weights of the individual concepts against each other
    )
    semantic_image = out.images[0]
    # semantic_image.show()
    return semantic_image


def upscaler(low_res_img, simple_prompt):
    upscaled_image = pipeline(prompt=simple_prompt, image=low_res_img).images[0]
    return upscaled_image


# Semantic guidance image generation using SD_V1.5 + Upscaler
semantic_prompt = ["family home",
                   "dogs sleeping",
                   "birthday",
                   "party"]

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
    # Generate
    semantic_image = semantic_guidance(prompt, semantic_prompt)
    result_img_path = f"presentation/semantic_sd15/semantic_sd15_image_{i}.png"
    semantic_image.save(result_img_path)

    # Upscale
    upscaled_image = upscaler(semantic_image, prompt)
    result_img_path = f"presentation/semantic_sd15/upscaled/upscaled_semantic_sd15_image_{i}.png"
    upscaled_image.save(result_img_path)

