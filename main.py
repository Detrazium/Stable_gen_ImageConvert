import os, logging
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from transformers import pipeline
from diffusers import (
ControlNetModel,
StableDiffusionControlNetImg2ImgPipeline,
DDIMScheduler,)
from diffusers.utils import load_image, make_image_grid
import torch
import numpy as np

imgMy = load_image(r"C:\Datasets\oJpmaLR37vo.jpg")

items = {'model': "Lykon/DreamShaper",
		 'controller': "lllyasviel/sd-controlnet-depth",
		 'ip_adapter': 'ip-adapter-plus-face_sd15.bin'}
prompt = 'masterpiece, portrait of a person, anime style, high quality, RAW photo, 8k uhd'

bad_prompt = 'monochrome, lowres, bad anatomy, worst quality, low quality, blurry'

def get_map_depth(imgMy, depth_ester):
	img = depth_ester(imgMy)['depth']
	img = np.array(img)
	img = img[:, :, None]
	img = np.concatenate([img, img, img], axis=2)
	detecter = torch.from_numpy(img).float() / 255.0
	depth_map = detecter.permute(2, 0, 1)
	return depth_map

depth_ester = pipeline('depth-estimation')
depth_image = get_map_depth(imgMy, depth_ester).unsqueeze(0).half()

controlnet = ControlNetModel.from_pretrained(items['controller'])
pipline = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(items['model'],controlnet=controlnet)
pipline.scheduler = DDIMScheduler.from_config(pipline.scheduler.config)

pipline.load_ip_adapter("h94/IP-Adapter", subfolder='models', weight_name=items['ip_adapter'])
pipline.set_ip_adapter_scale(0.6)

generatro = torch.Generator(device='cpu').manual_seed(33)
outer = pipline(prompt=prompt,
				negative_prompt=bad_prompt,
				image=imgMy,
				ip_adapter_image=imgMy,
				control_image = depth_image,
				generator=generatro).images[0]
item = make_image_grid([imgMy, outer], rows=1, cols=2)
item.save('images/imagereD2.png')

