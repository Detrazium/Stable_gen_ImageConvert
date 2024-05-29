from diffusion_stable import Stable_patterns_metod_v1
from diffusers.utils import load_image

prompt = "masterpiece, portrait of a person, anime style, high quality, RAW photo, 8k uhd"
bad_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality, blurry"
def main():
    Stable_patterns_metod_v1().gen_start(images=load_image(image), prompt=prompt, bad_prompt=bad_prompt)

if __name__ == '__main__':
    main()
