
from diffusers import StableDiffusionPipeline, DDIMScheduler

import torch
from PIL import Image

import config as cfg
from ip_adapter.ip_adapter import IPAdapter

device = "cuda"

pipe = StableDiffusionPipeline.from_single_file(cfg.sd15_base_model_path, torch_dtype=torch.float16)
pipe.safety_checker = None
pipe.feature_extractor = None
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.to(device)

image1 = Image.open("assets/input_venere.jpg")
image2 = Image.open("assets/input_portrait.jpg")

ip_adapter = IPAdapter(pipe, cfg.ipadapter_sd15_plus_path, cfg.image_encoder_sd15_path, device=device)

generator = torch.Generator().manual_seed(1)

"""
Plus model with one reference image and text prompt
"""
prompt_embeds, negative_prompt_embeds = ip_adapter.get_prompt_embeds(
    image1,
    prompt="beautiful renaissance woman",
    negative_prompt="blurry, horror, worst quality, low quality",
)

image = pipe(
    prompt_embeds=prompt_embeds,
    negative_prompt_embeds=negative_prompt_embeds,
    num_inference_steps=30,
    guidance_scale=5.0,
    num_images_per_prompt=1,
    height=512,
    width=512,
    generator=generator,
).images[0]
image.save("output/ipadapter_plus_sd15.webp", lossless=True, quality=100)

"""
Plus model with two reference images and text prompt
"""
prompt_embeds, negative_prompt_embeds = ip_adapter.get_prompt_embeds(
    [image1, image2],
    prompt="beautiful renaissance woman",
    negative_prompt="blurry, horror, worst quality, low quality",
)

image = pipe(
    prompt_embeds=prompt_embeds,
    negative_prompt_embeds=negative_prompt_embeds,
    num_inference_steps=30,
    guidance_scale=5.0,
    num_images_per_prompt=1,
    height=512,
    width=512,
    generator=generator,
).images[0]
image.save("output/ipadapter_plus_sd15_multi.webp", lossless=True, quality=100)

"""
Plus model with three reference images and noisy negative images
Negative image can be anything but it seems to react better to very noisy images
"""
noise = Image.effect_noise((224, 224), 1)
prompt_embeds, negative_prompt_embeds = ip_adapter.get_prompt_embeds(
    [image1, image2],
    prompt="beautiful renaissance woman",
    negative_prompt="blurry, horror, worst quality, low quality",
    negative_images=[noise, noise],
)

image = pipe(
    prompt_embeds=prompt_embeds,
    negative_prompt_embeds=negative_prompt_embeds,
    num_inference_steps=30,
    guidance_scale=5.0,
    num_images_per_prompt=1,
    height=512,
    width=512,
    generator=generator,
).images[0]
image.save("output/ipadapter_plus_sd15_noise.webp", lossless=True, quality=100)

"""
Plus model with one reference image and lower scale to give text more strength
"""
prompt_embeds, negative_prompt_embeds = ip_adapter.get_prompt_embeds(
    image1,
    prompt="beautiful renaissance woman wearing sunglasses",
    negative_prompt="blurry, horror, worst quality, low quality",
)
ip_adapter.set_scale(.6)

image = pipe(
    prompt_embeds=prompt_embeds,
    negative_prompt_embeds=negative_prompt_embeds,
    num_inference_steps=30,
    guidance_scale=5.0,
    num_images_per_prompt=1,
    height=512,
    width=512,
    generator=generator,
).images[0]
image.save("output/ipadapter_plus_sd15_text.webp", lossless=True, quality=100)
