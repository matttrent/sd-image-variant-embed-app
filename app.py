import gradio as gr
import torch
from PIL import Image

from diffusers import StableDiffusionImageVariationEmbedsPipeline

def main(
    input_im,
    base_prompt=None,
    edit_prompt=None,
    edit_prompt_weight=1.0,
    scale=3.0,
    steps=25,
    seed=0,
    ):

    generator = torch.Generator(device=device).manual_seed(int(seed))

    n_samples = 1
    images_list = pipe(
        n_samples*[input_im],
        base_prompt=base_prompt,
        edit_prompt=edit_prompt,
        edit_prompt_weight=edit_prompt_weight,
        guidance_scale=scale,
        num_inference_steps=steps,
        generator=generator,
        )

    return images_list.images

    images = []
    for i, image in enumerate(images_list.images):
        if(images_list["nsfw_content_detected"][i]):
            safe_image = Image.open(r"unsafe.png")
            images.append(safe_image)
        else:
            images.append(image)
    return images


description = \
"""
Generate variations on an input image using a fine-tuned version of Stable Diffision.
Trained by [Justin Pinkney](https://www.justinpinkney.com) ([@Buntworthy](https://twitter.com/Buntworthy)) at [Lambda](https://lambdalabs.com/)

This version has been ported to 🤗 Diffusers library, see more details on how to use this version in the [Lambda Diffusers repo](https://github.com/LambdaLabsML/lambda-diffusers).
__For the original training code see [this repo](https://github.com/justinpinkney/stable-diffusion).

![](https://raw.githubusercontent.com/justinpinkney/stable-diffusion/main/assets/im-vars-thin.jpg)

"""

article = \
"""
## How does this work?

The normal Stable Diffusion model is trained to be conditioned on text input. This version has had the original text encoder (from CLIP) removed, and replaced with
the CLIP _image_ encoder instead. So instead of generating images based a text input, images are generated to match CLIP's embedding of the image.
This creates images which have the same rough style and content, but different details, in particular the composition is generally quite different.
This is a totally different approach to the img2img script of the original Stable Diffusion and gives very different results.

The model was fine tuned on the [LAION aethetics v2 6+ dataset](https://laion.ai/blog/laion-aesthetics/) to accept the new conditioning.
Training was done on 4xA6000 GPUs on [Lambda GPU Cloud](https://lambdalabs.com/service/gpu-cloud).
More details on the method and training will come in a future blog post.
"""

device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionImageVariationEmbedsPipeline.from_pretrained(
    "matttrent/sd-image-variations-diffusers",
)
pipe = pipe.to(device)

def dummy(images, **kwargs):
    return images, False * len(images)

pipe.safety_checker = dummy

inputs = [
    gr.Image(),
    gr.Textbox(label="Base prompt"),
    gr.Textbox(label="Edit prompt"),
    gr.Slider(0.0, 2.0, value=1.0, step=0.1, label="Edit prompt weight"),
    gr.Slider(0, 25, value=3, step=1, label="Guidance scale"),
    gr.Slider(5, 50, value=25, step=5, label="Steps"),
    gr.Number(0, labal="Seed", precision=0)
]
output = gr.Gallery(label="Generated variations")
output.style(grid=2)

examples = [
    ["examples/painted ladies.png", None, None, 1.0, 3, 25, 0],
    ["examples/painted ladies.png", "a color photograph", "a black and white photograph", 1.0, 3, 25, 0],
    ["examples/painted ladies.png", "a color photograph", "a brightly colored oil painting", 1.0, 3, 25, 0],
]

demo = gr.Interface(
    fn=main,
    title="Stable Diffusion Image Variations",
    description=description,
    article=article,
    inputs=inputs,
    outputs=output,
    examples=examples,
    )
demo.launch()
