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

if __name__ == "__main__":
    # Load a test image
    test_image_path = r"C:\Users\joshu\PycharmProjects\CS5404-Final-Project\stable-point-aware-3d\demo_files\examples\fish.png"  # replace with your test image path
    img = Image.open(test_image_path)

    # Define some distortion levels to test
    distortions = [
        {"blur": 0, "noise": 0, "exposure": 1.0},
        {"blur": 10, "noise": 0, "exposure": 1.0},
        {"blur": 0, "noise": 150, "exposure": 1.0},
        {"blur": 0, "noise": 0, "exposure": 7.0},
    ]

    for i, params in enumerate(distortions):
        distorted = distort_image(img, **params)
        distorted.show(title=f"Distortion {i}")  # opens image
        # distorted.save(f"distorted_{i}.png")     # saves to file
        # print(f"[INFO] Saved distorted_{i}.png with params: {params}")