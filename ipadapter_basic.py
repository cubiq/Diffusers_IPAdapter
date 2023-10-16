
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
image3 = Image.open("assets/input_warrior.jpg")

ip_adapter = IPAdapter(pipe, cfg.ipadapter_sd15_path, cfg.image_encoder_sd15_path, device=device)

generator = torch.Generator().manual_seed(1)

"""
Base model with one reference image and text prompt
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
image.save("output/ipadapter_sd15.webp", lossless=True, quality=100)

"""
Base model with three reference images and text prompt
"""
prompt_embeds, negative_prompt_embeds = ip_adapter.get_prompt_embeds(
    [image1, image2, image3],
    prompt="beautiful renaissance warrior woman",
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
image.save("output/ipadapter_sd15_multi.webp", lossless=True, quality=100)

"""
Base model with three reference images and noisy negative images
Negative image can be anything but it seems to react better to very noisy images
"""
noise = Image.effect_noise((224, 224), 10)
prompt_embeds, negative_prompt_embeds = ip_adapter.get_prompt_embeds(
    [image1, image2, image3],
    prompt="beautiful renaissance warrior woman",
    negative_prompt="blurry, horror, worst quality, low quality",
    negative_images=[noise, noise, noise],
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
image.save("output/ipadapter_sd15_noise.webp", lossless=True, quality=100)

"""
Base model with three reference images and mandelbrot negative images
As an example I'm sending a mandelbrot as negative with often surprising results
"""
noise = Image.effect_mandelbrot((224, 224), (-3, -2.5, 2, 2.5), 100)
prompt_embeds, negative_prompt_embeds = ip_adapter.get_prompt_embeds(
    [image1, image2, image3],
    prompt="beautiful renaissance warrior woman",
    negative_prompt="blurry, horror, worst quality, low quality",
    negative_images=[noise, noise, noise],
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
image.save("output/ipadapter_sd15_madelbrot.webp", lossless=True, quality=100)

"""
Base model with one reference image and lower scale to give text more strength
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
image.save("output/ipadapter_sd15_text.webp", lossless=True, quality=100)

"""
Base model with two weighted reference images
"""
prompt_embeds, negative_prompt_embeds = ip_adapter.get_prompt_embeds(
    [image1, image2],
    prompt="beautiful renaissance woman",
    negative_prompt="blurry, horror, worst quality, low quality",
    weight=[1.0, .7],
)
ip_adapter.set_scale(1.0)
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
image.save("output/ipadapter_sd15_weight1.webp", lossless=True, quality=100)

"""
Base model with two weighted reference images
"""
prompt_embeds, negative_prompt_embeds = ip_adapter.get_prompt_embeds(
    [image1, image2],
    prompt="beautiful renaissance woman",
    negative_prompt="blurry, horror, worst quality, low quality",
    weight=[.7, 1.0],
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
image.save("output/ipadapter_sd15_weight2.webp", lossless=True, quality=100)
