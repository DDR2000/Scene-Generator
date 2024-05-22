import cv2
from PIL import Image
import numpy as np
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import torch
from diffusers import UniPCMultistepScheduler
import os
# Controlnet image+text+canny conditioning on sd1.5 images

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
pipe.enable_xformers_memory_efficient_attention()

for i in range(9):
    directory_path = "presentation/semantic_sd15"
    image_path = f"presentation/semantic_sd15/semantic_sd15_image_{i}.png"
    generated_path = f"presentation/semantic_sd15/control_net"
    create_directory(generated_path)
    image = Image.open(image_path)
    image = np.array(image)

    low_threshold = 100
    high_threshold = 200

    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    # canny_image.show()

    prompt = ", best quality, extremely detailed"
    prompt = [t + prompt for t in ["Boy", "Girl", "Man", "Women"]]
    generator = [torch.Generator(device="cpu").manual_seed(2) for i in range(len(prompt))]

    output = pipe(
        prompt,
        canny_image,
        negative_prompt=["monochrome, lowres, bad anatomy, worst quality, low quality"] * 4,
        num_inference_steps=20,
        generator=generator)
    cnt = 0
    for condition in ["Boy", "Girl", "Man", "Women"]:

        pil_image = output.images[cnt]

        # Specify the file path where you want to save the image
        file_path = f"{generated_path}/cnet_generated_image_{i}_{condition}.png"  # You can change the file extension based on the image format

        # Save the image
        pil_image.save(file_path)
        cnt += 1


