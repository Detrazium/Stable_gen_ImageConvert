import os, logging

import torch

logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
from transformers import pipeline
from PIL import Image
import numpy as np

class stt():
    def __init__(self):
        self.controlnetURL = "lllyasviel/sd-controlnet-depth"
        self.ipadapter = 'ip-adapter-plus-face_sd15.bin'
        self.modelLycons = "Lykon/DreamShaper"
        self.loadd_model()
    def loadd_model(self):
        controlnet = ControlNetModel.from_pretrained(self.controlnetURL, torch_dtype=torch.float32)
        pipline = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(self.modelLycons,
                                                                            controlnet=controlnet,
                                                                           torch_dtype=torch.float32).to('cpu')
        pipline.enable_vae_tiling()
        pipline.load_ip_adapter("h94/IP-Adapter",
                                 subfolder='models',
                                 weight_name=self.ipadapter)
        pipline.set_ip_adapter_scale(0.6)
        self.pipeline = pipline
    def depth_map(self, img):
        depthest = pipeline('depth-estimation')

        img = depthest(img)['depth']
        img = np.array(img)
        img = img[:, :, None]
        img = np.concatenate([img, img, img], axis=2)
        depthmap = Image.fromarray(img)
        return depthmap

    def start(self, image = None, prompt=None):
        self.img = load_image(image)
        self.control_image = self.depth_map(self.img)

        generator = torch.manual_seed(303)
        image = self.pipeline(
            image=self.img,
            prompt=prompt,
            ip_adapter_image=self.img,
            generator=generator,
            control_image = self.control_image,
            num_inference_steps=50
        ).images[0]

        image.save('n.png')
        return image

def main():
    stt()
if __name__ == '__main__':
    main()

