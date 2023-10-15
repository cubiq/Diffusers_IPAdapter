
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, ControlNetModel, StableDiffusionControlNetPipeline
from diffusers import DDIMScheduler

import torch
from PIL import Image

from libs.ip_adapter.ip_adapter import IPAdapter

base_model_path = "/mnt/ssd2/sd/models/sd15/absolutereality_v181.safetensors"
#base_model_path = "/mnt/ssd2/sd/models/sdxl/sd_xl_base_1.0_0.9vae.safetensors"
image_encoder_path = "models/clip_vision/ipadapter/"
ipadapter_model_path = "/home/matteo/bin/ComfyUI/custom_nodes/ComfyUI_IPAdapter_plus/models/ip-adapter_sd15.bin"
#ipadapter_model_path = "/home/matteo/bin/ComfyUI/custom_nodes/ComfyUI_IPAdapter_plus/models/ip-adapter-plus_sdxl_vit-h.bin"
device = "cuda"

#pipe = StableDiffusionPipeline.from_single_file(base_model_path, torch_dtype=torch.float16)
#pipe = StableDiffusionXLPipeline.from_single_file(base_model_path, torch_dtype=torch.float16)

controlnet = ControlNetModel.from_single_file("/home/matteo/bin/ComfyUI/models/controlnet/control_v11p_sd15s2_lineart_anime.pth")
lineart = Image.open("/home/matteo/bin/ComfyUI/input/manga.jpg")
pipe = StableDiffusionControlNetPipeline.from_single_file(base_model_path, controlnet=controlnet, torch_dtype=torch.float16)

pipe.watermark = None
pipe.safety_checker = None
pipe.feature_extractor = None
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.to(device)

image1 = Image.open("/home/matteo/bin/ComfyUI/input/venere_opti.png")
image2 = Image.open("/home/matteo/bin/ComfyUI/input/cwf_portrait.jpg")
image3 = Image.open("/home/matteo/bin/ComfyUI/input/woman.png")

ip_adapter = IPAdapter(pipe, ipadapter_model_path, image_encoder_path, device=device)

#noise = Image.effect_noise((224, 224), 1)
#noise = Image.effect_mandelbrot((224, 224), (-3, -2.5, 2, 2.5), 100)
#noise.save("mandelbrot.webp", lossless=True, quality=100)
#prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = ip_adapter.get_prompt_embeds(
prompt_embeds, negative_prompt_embeds = ip_adapter.get_prompt_embeds(
    [image1, image2, image3],
    #image1,
    prompt="beautiful renaissance warrior woman",
    negative_prompt="blurry, horror, worst quality, low quality",
#    negative_images=[noise, noise, noise],
)

ip_adapter.set_scale(1.0)

generator = torch.Generator().manual_seed(12345)

image = pipe(
    prompt_embeds=prompt_embeds,
    negative_prompt_embeds=negative_prompt_embeds,
    #pooled_prompt_embeds=pooled_prompt_embeds,
    #negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
    num_inference_steps=30,
    guidance_scale=5.0,
    num_images_per_prompt=1,
    #height=512,
    #width=512,
    generator=generator,
    image=lineart,
)

image.images[0].save("ipadapter_controlnet.webp", lossless=True, quality=100)
