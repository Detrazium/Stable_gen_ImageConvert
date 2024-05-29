from diffusion_stable import Stable_patterns_metod_v1
from diffusers.utils import load_image

image = 'https://s565vla.storage.yandex.net/rdisk/75cd38644df97568bd0d17a7dfff0661f7d3b7ad25230f2e638cb7765e5de880/664cc783/z2Hg9rf1o6KeuJ7Nd_tI11Ucr2_IGd7VQopCEeOZheReHV5r3AaW2bzTJxdROH5vu-w9jIL9uEQ6UQ-U-o4Z4A==?uid=0&filename=Image_.jpg&disposition=inline&hash=&limit=0&content_type=image%2Fjpeg&owner_uid=0&fsize=191565&hid=7d1fdb13425cb90c9a42494da164a660&media_type=image&tknv=v2&etag=abd8ea2d21daebe70dd23d95055c5d6b&ts=618f90f4ea6c0&s=81cb85b11b1e31e2ea43d3259639bb9d887d2c8924a7d0d092830528374ae410&pb=U2FsdGVkX18y7sfDPHoLL10vxFpZeNT4Jy2D_gmjCoDwAiS-PETaoY7QRQFjgK7BdBrMNXALd5W75kp0KOavVzp609hlZXZhclaVP5GviSc'
prompt = "masterpiece, portrait of a person, anime style, high quality, RAW photo, 8k uhd"
bad_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality, blurry"
def main():
    Stable_patterns_metod_v1().gen_start(images=load_image(image), prompt=prompt, bad_prompt=bad_prompt)

if __name__ == '__main__':
    main()
