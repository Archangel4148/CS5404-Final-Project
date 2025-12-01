import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

def distort_image(img: Image.Image, blur=0, noise=0, exposure=1.0):
    """Apply blur, noise, exposure shifts. Returns new PIL image."""

    # blur
    if blur > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur))

    # exposure
    if exposure != 1.0:
        img = ImageEnhance.Brightness(img).enhance(exposure)

    # noise
    if noise > 0:
        arr = np.array(img).astype(np.float32)
        noise_arr = np.random.normal(0, noise, arr.shape)
        arr = np.clip(arr + noise_arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    return img
