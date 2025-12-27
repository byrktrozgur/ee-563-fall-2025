import torch
import gradio as gr
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image, ImageFilter

DEVICE = "cuda"

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "sd-legacy/stable-diffusion-inpainting",
    torch_dtype=torch.float16
).to(DEVICE)

def inpaint_only(editor_data, prompt, negative_prompt, cfg, steps):
    # ---- Extract image & mask ----
    image = editor_data["background"].convert("RGB")
    mask = editor_data["layers"][0]

    # ---- Enforce valid size (divisible by 8) ----
    w, h = image.size
    w, h = w - w % 8, h - h % 8
    image = image.resize((w, h))
    mask = mask.resize((w, h))

    # ---- Prepare mask (VERY IMPORTANT) ----
    mask = mask.convert("L")
    mask = mask.filter(ImageFilter.GaussianBlur(4))  # smooth edges
    mask = Image.eval(mask, lambda x: 255 if x > 0 else 0)
    mask = mask.convert("RGB")

    # ---- Inpainting ----
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        mask_image=mask,
        guidance_scale=cfg,
        num_inference_steps=steps
    ).images[0]

    return result


with gr.Blocks() as demo:
    gr.Markdown("# Stable Diffusion – Inpainting Only (Photo Restoration)")

    editor = gr.ImageEditor(
        label="Upload image and paint ONLY damaged areas",
        type="pil",
        brush=gr.Brush(colors=["#FFFFFF"], default_size=20),
        canvas_size=(512, 512)
    )

    prompt = gr.Textbox(
        value="restored vintage photograph, realistic, minimal changes, preserve identity",
        label="Prompt"
    )

    negative_prompt = gr.Textbox(
        value="cartoon, painting, oversharpened, distorted face, artificial colors",
        label="Negative Prompt"
    )

    cfg = gr.Slider(
        minimum=3.0,
        maximum=8.0,
        value=5.5,
        step=0.5,
        label="CFG (Guidance Scale)"
    )

    steps = gr.Slider(
        minimum=20,
        maximum=60,
        value=40,
        step=5,
        label="Inference Steps"
    )

    output = gr.Image(type="pil", label="Restored Output")

    gr.Button("Inpaint").click(
        fn=inpaint_only,
        inputs=[editor, prompt, negative_prompt, cfg, steps],
        outputs=output
    )

demo.launch()
