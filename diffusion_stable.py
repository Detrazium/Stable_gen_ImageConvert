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



class Stable_patterns_metod_v1():
	def __init__(self, image=None):
		self.modelLycon = "Lykon/DreamShaper"
		self.contNet_depth = "lllyasviel/sd-controlnet-depth"
		self.face_sd15Adapt = 'ip-adapter-plus-face_sd15.bin'
		self.image = self.get_image(image)
	def get_image(self, img):
		if img == None:
			raise ValueError('None image, please get image')
		img = load_image(img)
		return img
	def create_image(self, prompt, bad_prompt):
		imgMy = self.image
		def get_map_depth(imgMy, depth_ester):
			img = depth_ester(imgMy)['depth']
			img = np.array(img)
			img = img[:, :, None]
			img = np.concatenate([img, img, img], axis=2)
			detecter = torch.from_numpy(img).float() / 255.0
			depth_map = detecter.permute(2, 0, 1)
			return depth_map
		depth_ester = pipeline('depth-estimation')
		depth_image = get_map_depth(self.image, depth_ester).unsqueeze(0).half()
		controlnet = ControlNetModel.from_pretrained(self.contNet_depth)
		pipline = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(self.modelLycon,controlnet=controlnet)
		pipline.scheduler = DDIMScheduler.from_config(pipline.scheduler.config)
		pipline.load_ip_adapter("h94/IP-Adapter", subfolder='models', weight_name=self.face_sd15Adapt)
		pipline.set_ip_adapter_scale(0.6)
		generatro = torch.Generator(device='cpu').manual_seed(33)
		outer = pipline(prompt=prompt,
						negative_prompt=bad_prompt,
						image=imgMy,
						ip_adapter_image=imgMy,
						control_image = depth_image,
						generator=generatro).images[0]
		item = make_image_grid([imgMy, outer], rows=1, cols=2)
		return item

def main():
	Stable_patterns_metod_v1
if __name__ == '__main__':
	main()

