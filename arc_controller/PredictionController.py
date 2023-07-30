import json
from base64 import b64decode
from io import BytesIO
from PIL import Image, ImageOps, ImageFilter
import numpy as np
from scipy import ndimage
import torch
from torchvision import transforms
import arc_model.CharacterModel as characterModel
import torch.nn.functional as F
import pandas as pd
import os
import random
import string

df = pd.read_csv('./ds_train/CharacterMapping.csv', header=0)
classesList = df["Character"].tolist()

def url_to_img(dataURL):
    # Convert dataURL to string
    string = str(dataURL)
    # Find the index of the comma
    comma = string.find(",")
    # Extract the image code by slicing the string
    code = string[comma + 1:]
    # Decode the image code using Base64 decoding
    decoded = b64decode(code)
    # Create a buffer and store the decoded code as binary data
    buf = BytesIO(decoded)
    # Open the image using PIL's Image.open() function
    img = Image.open(buf)

    # Convert the image to grayscale with an alpha channel ("LA" mode)
    converted = img.convert("LA")
    # Set pixels with alpha value 0 (transparent) to white (255, 255)
    la = np.array(converted)
    la[la[..., -1] == 0] = [255, 255]
    whiteBG = Image.fromarray(la)

    # Convert the image to grayscale ("L" mode)
    converted = whiteBG.convert("L")
    # Invert the grayscale image
    inverted = ImageOps.invert(converted)

    # Get the bounding box of the inverted image
    bounding_box = inverted.getbbox()
    # Pad the bounding box by 5 pixels in each direction
    padded_box = tuple(map(lambda i, j: i + j, bounding_box, (-5, -5, 5, 5)))
    # Crop the inverted image using the padded bounding box
    cropped = inverted.crop(padded_box)

    # Apply a maximum filter to the cropped image with a size of 5 pixels
    thick = cropped.filter(ImageFilter.MaxFilter(5))

    # Resize the thick image to a target size of 48x48 pixels using Lanczos resampling
    ratio = 48.0 / max(thick.size)
    new_size = tuple([int(round(x * ratio)) for x in thick.size])
    res = thick.resize(new_size, Image.LANCZOS)

    # Convert the resized image to a NumPy array
    arr = np.asarray(res)
    # Compute the center of mass of the array using scipy's center_of_mass() function
    com = ndimage.measurements.center_of_mass(arr)
    # Create a new grayscale image with a size of 64x64 pixels
    result = Image.new("L", (64, 64))
    # Compute the box coordinates to paste the resized image onto the new image
    box = (int(round(32.0 - com[1])), int(round(32.0 - com[0])))
    # Paste the resized image onto the new image at the computed box coordinates
    result.paste(res, box)

    # Generate a random file name
    # Generate a random file name with text and numbers
    characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    file_name_non_processed = "white_" + ''.join(random.choice(characters) for _ in range(8)) + ".png"
    file_name = ''.join(random.choice(characters) for _ in range(8)) + ".png"
    # Specify the folder path to save the image
    folder_path = "generated/canvas_drawings/"
    # Create the folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)
    # Save the image to the specified folder with the random file name
    file_path_non_processed = os.path.join(folder_path, file_name_non_processed)
    file_path = os.path.join(folder_path, file_name)
    #converted.save(file_path_non_processed)
    result.save(file_path)
    print("Canvas processed image saved to:", file_path)

    # Return the final result image
    return result


def transformImg(img):
    my_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    return my_transforms(img).unsqueeze(0)


def get_prediction(url, net):
    img = url_to_img(url)
    transformed = transformImg(img)
    output = net(transformed)
    probabilities, predictions = torch.topk(output.data, 1)

    print("prediction - ", classesList[predictions.data[0][0]], " - ",
          predictions.data[0][0].item(), " : Probability - ", probabilities.data[0][0].item())

    confidence1 = int(round(probabilities.data[0][0].item() * 100))
    # confidence2 = int(round(probabilities.data[0][1].item() * 100))
    guess = characterModel.Character(predictions.data[0][0].item(),
                                     classesList[predictions.data[0][0]],
                                     confidence1)

    return json.dumps(guess.__dict__)