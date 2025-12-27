from diffusers import StableDiffusionInpaintPipeline
import torch  
# from PIL import Image

# MODEL_ID = "CompVis/stable-diffusion-v1-4"
MODEL_ID = "sd-legacy/stable-diffusion-inpainting"

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    MODEL_ID,
    # revision="fp16",
    torch_dtype=torch.float16,
    use_auth_token=True
).to("cuda")

# prompt = "Face of a yellow cat, high resolution, sitting on a park bench"
# #image and mask_image should be PIL images.
# #The mask structure is white for inpainting and black for keeping as is

# image = pipe(prompt=prompt, image=image, mask_image=mask_image).images[0]
# image.save("./yellow_cat_on_park_bench.png")
