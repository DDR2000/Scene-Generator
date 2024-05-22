from diffusers.utils import load_image
import torch
import os
from diffusers import StableUnCLIPImg2ImgPipeline


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


create_directory("presentation/sdxl/reimagine")
create_directory("presentation/photo_sdxl/reimagine")

reimagine_pipe = StableUnCLIPImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-unclip",
                                                                 torch_dtype=torch.float16, variation="fp16")
reimagine_pipe.to("cuda")


def reimagine(image_path):  # unclip (reimagine)
    # get image
    init_image = load_image(image_path)
    # run image variation
    images = reimagine_pipe(init_image).images
    return images


for i in range(9):
    filename = f"presentation/sdxl/sdxl_image_{i}.png"
    images = reimagine(filename)
    images[0].save(f"presentation/sdxl/reimagine/reimagine_image_{i}.png")

    filename = f"presentation/photo_sdxl/sdxl_photo_image_{i}.png"
    images = reimagine(filename)
    images[0].save(f"presentation/photo_sdxl/reimagine/photo_reimagine_image_{i}.png")

