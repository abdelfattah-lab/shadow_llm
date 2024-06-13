import os
from PIL import Image, ImageOps

def crop_whitespace(image):
    grayscale = image.convert("L")  # Convert image to grayscale
    inverted_image = ImageOps.invert(grayscale)  # Invert the image to make the whitespace black
    bbox = inverted_image.getbbox()  # Get the bounding box of the non-black (originally non-white) areas
    return image.crop(bbox)

def process_images(input_folder):
    files = sorted([f for f in os.listdir(input_folder) if f.endswith('.png')])
    for file in files:
        img = Image.open(os.path.join(input_folder, file))
        img = crop_whitespace(img)
        img.save(os.path.join(input_folder, file))

input_folder = './tograph'
process_images(input_folder)
