import os, logging
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from transformers import pipeline
from diffusers import (
ControlNetModel,
StableDiffusionControlNetImg2ImgPipeline,
DDIMScheduler,)
from diffusers.utils import make_image_grid, load_image
import torch
import numpy as np

class Stable_patterns_metod_v1():
	def __init__(self):
		self.modelLycon = "Lykon/DreamShaper"
		self.contNet_depth = "lllyasviel/sd-controlnet-depth"
		self.face_sd15Adapt = 'ip-adapter-plus-face_sd15.bin'
		self.get_model()
	def get_model(self):
		controlnet = ControlNetModel.from_pretrained(self.contNet_depth)
		pipline = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(self.modelLycon,
																		   controlnet=controlnet)
		pipline.scheduler = DDIMScheduler.from_config(pipline.scheduler.config)
		pipline.load_ip_adapter("h94/IP-Adapter",
								subfolder='models',
								weight_name=self.face_sd15Adapt)
		pipline.set_ip_adapter_scale(0.6)

		self.pipline = pipline

	def create_image(self, imager):
		self.imgMy = imager
		def get_map_depth(imgMy, depth_ester):
			img = depth_ester(imgMy)['depth']
			img = np.array(img)
			img = img[:, :, None]
			img = np.concatenate([img, img, img], axis=2)
			detecter = torch.from_numpy(img).float() / 255.0
			depth_map = detecter.permute(2, 0, 1)
			return depth_map
		depth_ester = pipeline('depth-estimation')
		self.depth_image = get_map_depth(self.imgMy, depth_ester).unsqueeze(0).half()
	def gen_start(self, images, prompt, bad_prompt):
		self.create_image(images)

		generatro = torch.Generator().manual_seed(33)
		outer = self.pipline(
			prompt=prompt,
			negative_prompt=bad_prompt,
			image=self.imgMy,
			ip_adapter_image=self.imgMy,
			control_image = self.depth_image,
			generator=generatro).images[0]
		item = make_image_grid([self.imgMy, outer], rows=1, cols=2)
		return item

def main():
	Stable_patterns_metod_v1()

if __name__ == '__main__':
	main()

