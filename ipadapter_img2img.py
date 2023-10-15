
from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler

import torch
from PIL import Image

import config as cfg
from ip_adapter.ip_adapter import IPAdapter

device = "cuda"

pipe = StableDiffusionImg2ImgPipeline.from_single_file(cfg.sd15_base_model_path, torch_dtype=torch.float16)
pipe.safety_checker = None
pipe.feature_extractor = None
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.to(device)

image1 = Image.open("assets/input_venere.jpg")
noise_image = Image.open("assets/input_portrait.jpg")

ip_adapter = IPAdapter(pipe, cfg.ipadapter_sd15_plus_path, cfg.image_encoder_sd15_path, device=device)

generator = torch.Generator().manual_seed(1)

"""
Image2Image
"""
prompt_embeds, negative_prompt_embeds = ip_adapter.get_prompt_embeds(
    image1,
    prompt="smiling woman",
    negative_prompt="blurry, horror, worst quality, low quality",
)
ip_adapter.set_scale(.7)
image = pipe(
    prompt_embeds=prompt_embeds,
    negative_prompt_embeds=negative_prompt_embeds,
    num_inference_steps=50,
    guidance_scale=6.0,
    num_images_per_prompt=1,
    generator=generator,
    strength=0.5,
    image=noise_image,
).images[0]
image.save("output/ipadapter_plus_sd15_i2i.webp", lossless=True, quality=100)

"""
Image2Image with noise negative image
"""
noise = Image.effect_noise((224, 224), 10)
prompt_embeds, negative_prompt_embeds = ip_adapter.get_prompt_embeds(
    image1,
    prompt="smiling woman",
    negative_prompt="blurry, horror, worst quality, low quality",
    negative_images=noise
)
ip_adapter.set_scale(.7)
image = pipe(
    prompt_embeds=prompt_embeds,
    negative_prompt_embeds=negative_prompt_embeds,
    num_inference_steps=50,
    guidance_scale=6.0,
    num_images_per_prompt=1,
    generator=generator,
    strength=0.5,
    image=noise_image,
).images[0]
image.save("output/ipadapter_plus_sd15_i2i_noise.webp", lossless=True, quality=100)