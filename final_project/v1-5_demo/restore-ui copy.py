import torch
import gradio as gr
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image, ImageStat, ImageFilter, ImageOps


# =========================
# Configuration
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "sd-legacy/stable-diffusion-inpainting",
    torch_dtype=DTYPE
).to(DEVICE)

pipe.enable_attention_slicing()

# Optional: modest speed/memory improvements on CUDA
if DEVICE == "cuda":
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

# =========================
# Helpers
# =========================
def _make_divisible_by_8(img: Image.Image) -> Image.Image:
    w, h = img.size
    w2, h2 = w - (w % 8), h - (h % 8)
    if (w2, h2) != (w, h):
        return img.resize((w2, h2), resample=Image.Resampling.LANCZOS)
    return img

def _prep_mask(mask: Image.Image, size, blur=6, threshold=8, feather=0):
    """
    White = area to modify
    Returns:
      mask_rgb: RGB
      mask_l:   L
    """
    mask = mask.resize(size, resample=Image.Resampling.NEAREST)
    mask_l = mask.convert("L")

    # Detect inverted mask (mostly white = likely inverted)
    stat = ImageStat.Stat(mask_l)
    if stat.mean[0] > 200:
        mask_l = ImageOps.invert(mask_l)

    # Blur to soften edges
    if blur > 0:
        mask_l = mask_l.filter(ImageFilter.GaussianBlur(blur))

    # Binarize
    mask_l = Image.eval(mask_l, lambda x: 255 if x > threshold else 0)

    # Optional feathering to hide seams
    if feather > 0:
        mask_l = mask_l.filter(ImageFilter.GaussianBlur(feather))

    return mask_l.convert("RGB"), mask_l


# =========================
# Core logic
# =========================
def inpaint_indoor_object_removal(
    editor_data,
    object_hint,
    negative_prompt,
    cfg,
    steps,
    strength,
    mask_blur,
    seam_feather,
    seed):

    if editor_data is None:
        return None

    image = editor_data["background"].convert("RGB")
    if not editor_data.get("layers"):
        return image

    mask = editor_data["layers"][0]

    # ---- enforce valid size (divisible by 8) ----
    image = _make_divisible_by_8(image)
    size = image.size

    # ---- prepare mask (indoor removal needs softer edges) ----
    mask_rgb, mask_l = _prep_mask(
        mask,
        size=size,
        blur=int(mask_blur),
        threshold=8,
        feather=int(seam_feather)
    )

    # ---- indoor object removal prompts ----
    # object_hint helps if you want: "remove chair", "remove lamp", etc.
    base_prompt = (
        "photorealistic indoor room, clean continuous background, "
        "natural continuation of walls and floor, consistent perspective, "
        "matching lighting and shadows, realistic textures, seamless inpainting"
    )
    if object_hint and object_hint.strip():
        prompt = f"{base_prompt}. Remove {object_hint.strip()}."
    else:
        prompt = base_prompt

    indoor_negative = (
        "text, watermark, logo, extra objects, duplicate objects, people, "
        "distortion, warped geometry, melted, blurry, low quality, artifacts, "
        "wrong perspective, inconsistent shadows, harsh edges, seams"
    )
    if negative_prompt and negative_prompt.strip():
        indoor_negative = f"{indoor_negative}, {negative_prompt.strip()}"

    # ---- deterministic seed ----
    generator = None
    if seed is not None and int(seed) >= 0:
        generator = torch.Generator(device=DEVICE).manual_seed(int(seed))

    # ---- inpainting ----
    # strength is supported by the inpaint pipeline (how strongly to alter the masked region).
    result = pipe(
        prompt=prompt,
        negative_prompt=indoor_negative,
        image=image,
        mask_image=mask_rgb,
        guidance_scale=float(cfg),
        num_inference_steps=int(steps),
        strength=float(strength),
        generator=generator
    ).images[0]

    # ---- SAFETY: force identical size + correct modes ----
    if result.size != image.size:
        result = result.resize(image.size, resample=Image.Resampling.LANCZOS)

    if mask_l.size != image.size:
        mask_l = mask_l.resize(image.size, resample=Image.Resampling.NEAREST)

    # Ensure modes
    image_rgb = image.convert("RGB")
    result_rgb = result.convert("RGB")
    mask_l = mask_l.convert("L")

    # Composite only where mask is white
    out = Image.composite(result_rgb, image_rgb, mask_l)
    return out



# =========================
# Gradio UI
# =========================
with gr.Blocks() as demo:
    gr.Markdown("# Stable Diffusion – Indoor Scene Object Removal")

    editor = gr.ImageEditor(
        label="Upload an indoor photo and paint the object to REMOVE",
        type="pil",
        brush=gr.Brush(colors=["#FFFFFF"], default_size=25),
        canvas_size=(768, 768)  # indoors often benefits from a bit more resolution
    )

    object_hint = gr.Textbox(
        label="Object hint (optional)",
        value="the chair",
        placeholder="e.g., the chair, the lamp, the TV, the table"
    )

    negative_prompt = gr.Textbox(
        label="Extra negative prompt (optional)",
        value=""
    )

    cfg = gr.Slider(3.0, 9.0, value=5.5, step=0.5, label="CFG Scale")
    steps = gr.Slider(20, 60, value=45, step=5, label="Inference Steps")

    strength = gr.Slider(
        0.4, 1.0, value=0.85, step=0.05,
        label="Strength (higher = stronger removal)"
    )

    mask_blur = gr.Slider(
        0, 20, value=8, step=1,
        label="Mask blur (helps hide seams)"
    )

    seam_feather = gr.Slider(
        0, 12, value=3, step=1,
        label="Extra feather (post-threshold blur)"
    )

    seed = gr.Number(
        value=-1,
        precision=0,
        label="Seed (-1 = random)"
    )

    output = gr.Image(type="pil", label="Output Image")

    gr.Button("Remove Object").click(
        fn=inpaint_indoor_object_removal,
        inputs=[editor, object_hint, negative_prompt, cfg, steps, strength, mask_blur, seam_feather, seed],
        outputs=output
    )

demo.launch()
