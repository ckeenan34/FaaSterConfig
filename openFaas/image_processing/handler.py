try:
    import Image, ImageFilter
except ImportError as e:
    from PIL import Image, ImageFilter

import requests
from io import BytesIO
import os
import base64
TMP = "/tmp/"


def download_image(url):
    response = requests.get(url)
    if response.status_code == 200:
        return Image.open(BytesIO(response.content))
    else:
        raise ValueError(f"Failed to download image from {url}. Status code: {response.status_code}")
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_image

def flip(image, file_name):
    path_list = []
    path = TMP + "flip-left-right-" + file_name
    img = image.transpose(Image.FLIP_LEFT_RIGHT)
    img.save(path)
    path_list.append(path)

    path = TMP + "flip-top-bottom-" + file_name
    img = image.transpose(Image.FLIP_TOP_BOTTOM)
    img.save(path)
    path_list.append(path)

    return path_list


def rotate(image, file_name):
    path_list = []
    path = TMP + "rotate-90-" + file_name
    img = image.transpose(Image.ROTATE_90)
    img.save(path)
    path_list.append(path)

    path = TMP + "rotate-180-" + file_name
    img = image.transpose(Image.ROTATE_180)
    img.save(path)
    path_list.append(path)

    path = TMP + "rotate-270-" + file_name
    img = image.transpose(Image.ROTATE_270)
    img.save(path)
    path_list.append(path)

    return path_list


def filter(image, file_name):
    path_list = []
    path = TMP + "blur-" + file_name
    img = image.filter(ImageFilter.BLUR)
    img.save(path)
    path_list.append(path)

    path = TMP + "contour-" + file_name
    img = image.filter(ImageFilter.CONTOUR)
    img.save(path)
    path_list.append(path)

    path = TMP + "sharpen-" + file_name
    img = image.filter(ImageFilter.SHARPEN)
    img.save(path)
    path_list.append(path)

    return path_list


def gray_scale(image, file_name):
    path = TMP + "gray-scale-" + file_name
    img = image.convert('L')
    img.save(path)
    return [path]


def resize(image, file_name):
    path = TMP + "resized-" + file_name
    image.thumbnail((128, 128))
    image.save(path)
    return [path]


def image_processing(image_url):
    image = download_image(image_url)
    path_list = []

    path_list += flip(image, os.path.basename(image_url))
    path_list += rotate(image, os.path.basename(image_url))
    path_list += filter(image, os.path.basename(image_url))
    path_list += gray_scale(image, os.path.basename(image_url))
    path_list += resize(image, os.path.basename(image_url))

    return path_list


def handle(event):
    image_url = event
    output_dir = "processed/"

    result_paths = image_processing(image_url)
    saved_paths = []

    for result_path in result_paths:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
        result_filename = os.path.basename(result_path)
        output_path = os.path.join(output_dir, result_filename)
        os.rename(result_path, output_path)
        encoded_image = encode_image(output_path)
        saved_paths.append(encoded_image)
    download_urls = [f"/images/{os.path.basename(path)}" for path in saved_paths]

    return {"message": "Image processing complete.", "download_urls": download_urls}


