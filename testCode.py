import os, logging
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from diffusers import ControlNetModel, UniPCMultistepScheduler, AutoPipelineForImage2Image, StableDiffusionImg2ImgPipeline, StableDiffusionControlNetImg2ImgPipeline
from diffusers.utils import load_image

new_item = 'restart'
imgMy = load_image(r"C:\Datasets\oJpmaLR37vo.jpg")
controler = "lllyasviel/sd-controlnet-depth"

piple3 = 'masterpiece, portrait of a person, anime style, high quality, RAW photo, 8k uhd'
piple5 = ('Black skull surrounded by swords, cover, victor, smoke, fate, art, high quality, fear, '
		  'halo of blades, emblem, fantasy, steel, necronomicon, a huge number of details, '
		  'ornate ornament, graphics, lines, a symbol, just a skull without a body, vector drawing, pathos, '
		  'a lot of smoke, absolute symmetry')

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth")
pipline = StableDiffusionControlNetImg2ImgPipeline.from_pretrained("Lykon/DreamShaper", controlnet=controlnet)
pipline.scheduler = UniPCMultistepScheduler.from_config(pipline.scheduler.config)
# generator = torch.Generator().manual_seed(8)

pipline.load_ip_adapter("h94/IP-Adapter", subfolder='models', weight_name='ip-adapter-plus-face_sd15.bin')
pipline.set_ip_adapter_scale(0.6)

image = pipline(prompt=piple3,
				ip_adapt_image=imgMy,
				negative_prompt='deformed, ugly, wrong proportion, low res, bad anatomy,',
				strength=0.75,
				guidance_scale=7.5,
				# generator = generator
				).images[0]
image.save('image6.png')

