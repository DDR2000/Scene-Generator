import torch
from diffusers import SemanticStableDiffusionPipeline

pipe = SemanticStableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

out = pipe(
    prompt="living room with furniture",
    num_images_per_prompt=1,
    guidance_scale=7,
    editing_prompt=[
        "journal table, coffee table, desk",  # Concepts to apply
        "sofa, couch, sectional",
        "flower pot, exotic plant",
        "TV, flat display, cinema projector",
    ],
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
    ],  # Threshold for each concept. Threshold equals the percentile of the latent space that will be discarded. I.e. threshold=0.99 uses 1% of the latent dimensions
    edit_momentum_scale=0.3,  # Momentum scale that will be added to the latent guidance
    edit_mom_beta=0.6,  # Momentum beta
    edit_weights=[1, 1, 1, 1, 1],  # Weights of the individual concepts against each other
)
image = out.images[0]
image.show()

# plus upscale
import requests
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionUpscalePipeline
import torch

# load model and scheduler
model_id = "stabilityai/stable-diffusion-x4-upscaler"
pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipeline = pipeline.to("cuda")

# let's download an  image
#url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd2-upscale/low_res_cat.png"
#response = requests.get(url)
#low_res_img = Image.open(BytesIO(response.content)).convert("RGB")
#low_res_img = low_res_img.resize((512, 512))
low_res_img = image

prompt = "living room"

upscaled_image = pipeline(prompt=prompt, image=low_res_img).images[0]
upscaled_image.save("living room.png")
