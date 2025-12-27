import torch
import gradio as gr
from demo import pipe

model_id = "sd-legacy/stable-diffusion-v1-5"

def generate_image(prompt):
    image = pipe(
        prompt,
        num_inference_steps=30,
        guidance_scale=7.5
    ).images[0]
    return image

demo = gr.Interface(
    fn=generate_image,
    inputs=gr.Textbox(
        label="Prompt",
        placeholder="a photo of an astronaut riding a horse on mars"
    ),
    outputs=gr.Image(type="pil"),
    title="Stable Diffusion v1.5 Demo"
)

demo.launch()
