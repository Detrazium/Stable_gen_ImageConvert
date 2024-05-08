image = 'imgs/imager.jpg'
prompt = 'masterpiece, portrait of a person, anime style, high quality, RAW photo, 8k uhd'
prompt2 = 'masterpiece, portrait of a person, anime style, high quality, RAW photo, 8k uhd, dragon, necronomicon, dragon man'
bad_prompt = 'monochrome, lowres, bad anatomy, worst quality, low quality, blurry'

from diffusion_stable import Stable_patterns_metod_v1

img = Stable_patterns_metod_v1(image).create_image(prompt=prompt2, bad_prompt=bad_prompt)
img.save('imgs/imager1.jpg')
