from diffusers import DiffusionPipeline
from diffusers.utils import load_image
test = 'test'

ell = torch.cuda.is_available()
ll = torch.version.cuda
print(ll)

print(ell)
pape = ('Divine serpent, colossus serpent, giant serpent, mystical serpent, '
		'absolute, serpent god, runes, fog, ultra graphics, cobra, '
		'serpent stands upright, magic, sorcerer serpent, '
		'blue-green scales, dragon serpent, fantasy, fantasy')
imgMy = load_image(r"C:\Datasets\oJpmaLR37vo.jpg")

pipeline = DiffusionPipeline.from_pretrained("Lykon/DreamShaper", torch_dtype=torch.float32)
pipeline.load_ip_adapter("h94/IP-Adapter", subfolder='models', weight_name='ip-adapter-plus-face_sd15.bin')
pipeline.set_ip_adapter_scale(0.6)
image = pipeline(prompt=pape, ip_adapter_image=imgMy).images[0]
image.save('imageR.png')

