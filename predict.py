from diffusion_stable import stt
from diffusers import StableDiffusionPipeline
from cog import BasePredictor, Input, Path
import tempfile


class Predictor(BasePredictor):
    def setup(self) -> None:
        StableDiffusionPipeline.from_pretrained("Lykon/DreamShaper")
        self.model = stt()

    def predict(
        self,
        image: Path = Input(description="input image"),
        prompt: str = Input(description='prompt')
    ) -> Path:
        itog_image = self.model.start(image=image, prompt=prompt)

        output = Path(tempfile.mkdtemp()) / 'itoger.png'
        itog_image.save(output)
        return Path(output)
