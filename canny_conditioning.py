from diffusers.utils import load_image
import torch
import os
import cv2
from PIL import Image
import numpy as np
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


create_directory("presentation/sdxl/canny_conditioning")
create_directory("presentation/photo_sdxl/canny_conditioning")

controlnet = ControlNetModel.from_pretrained(
        "diffusers/controlnet-canny-sdxl-1.0",
        torch_dtype=torch.float16
    )
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
conditioning_pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet=controlnet,
        vae=vae,
        torch_dtype=torch.float16,
    )
# pipe.inference_mode()
conditioning_pipe.enable_model_cpu_offload()


def control_net_conditioning(image_path, conditioning_prompt):
    negative_prompt = 'low quality, bad quality, sketches'
    init_image = load_image(image_path)
    controlnet_conditioning_scale = 0.5  # recommended for good generalization

    cond_image = np.array(init_image)
    cond_image = cv2.Canny(cond_image, 100, 200)
    cond_image = cond_image[:, :, None]
    cond_image_np = np.concatenate([cond_image, cond_image, cond_image], axis=2)
    cond_image = Image.fromarray(cond_image_np)
    # cond_image.show()
    images = conditioning_pipe(
        conditioning_prompt, negative_prompt=negative_prompt, image=cond_image,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
    ).images

    # images[0].show()
    return cond_image_np, images


# Generate control net conditioning image
conditioning_prompt = "thanksgiving family dinner"

for i in range(9):
    filename = f"presentation/sdxl/sdxl_image_{i}.png"
    cond_image_np, images = control_net_conditioning(filename, conditioning_prompt)
    result_img_path = f"presentation/sdxl/canny_conditioning/canny_conditioning_image_{i}.png"
    result_image_np = np.array(images[0])
    cond_result_img_path = f"presentation/sdxl/canny_conditioning/canny_image_{i}.png"
    cv2.imwrite(result_img_path, result_image_np)
    cv2.imwrite(cond_result_img_path, cond_image_np)

    filename = f"presentation/photo_sdxl/sdxl_photo_image_{i}.png"
    cond_image_np, images = control_net_conditioning(filename, conditioning_prompt)
    result_img_path = f"presentation/photo_sdxl/canny_conditioning/canny_conditioning_image_{i}.png"
    result_image_np = np.array(images[0])
    cond_result_img_path = f"presentation/photo_sdxl/canny_conditioning/canny_image_{i}.png"
    cv2.imwrite(result_img_path, result_image_np)
    cv2.imwrite(cond_result_img_path, cond_image_np)
