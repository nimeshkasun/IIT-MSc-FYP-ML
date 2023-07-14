import os
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from scipy.ndimage import center_of_mass
import torchvision.transforms as transforms

input_folder = 'C:/Users/iitfypvmadmin/Downloads/Dataset-Tamil/train-test-classwise/A'
output_folder = '../ds_source/tamil/test'

class Process(object):
    def __call__(self, img):
        # Convert the image to grayscale
        convertedImg = img.convert("L")
        # Invert the image (convert white pixels to black and black pixels to white)
        invertedImg = ImageOps.invert(convertedImg)
        # Apply Max filter to the inverted image (remove small noise and enhance the edges of the image)
        filteredImg = invertedImg.filter(ImageFilter.MaxFilter(5))
        # filteredImg = convertedImg.filter(ImageFilter.MaxFilter(5))
        # Resize the image to 48x48 using Lanczos interpolation
        resizeRatio = 48.0 / max(filteredImg.size)
        #resizeRatio = 48.0 / max(convertedImg.size)
        newSize = tuple([int(round(x * resizeRatio)) for x in filteredImg.size])
        #newSize = tuple([int(round(x * resizeRatio)) for x in convertedImg.size])
        resizeImg = filteredImg.resize(newSize, Image.LANCZOS)
        #resizeImg = convertedImg.resize(newSize, Image.LANCZOS)

        # Convert the resized image to a numpy array
        resizeImgArray = np.asarray(resizeImg)
        # Find the center of mass of the image
        com = center_of_mass(resizeImgArray)
        # Create a new image of size 64x64
        result = Image.new("L", (64, 64))
        # Calculate the top-left corner of the resized image in the new image
        box = (int(round(32.0 - com[1])), int(round(32.0 - com[0])))
        # Paste the resized image in the new image with calculated box coordinates
        result.paste(resizeImg, box)
        return result

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Iterate over all subfolders in the input folder
for class_folder in os.listdir(input_folder):
    class_folder_path = os.path.join(input_folder, class_folder)
    if not os.path.isdir(class_folder_path):
        continue

    # Create output class folder
    output_class_folder = os.path.join(output_folder, class_folder)
    if not os.path.exists(output_class_folder):
        os.makedirs(output_class_folder)

    # Iterate over all images in the class folder
    for image_file in os.listdir(class_folder_path):
        image_path = os.path.join(class_folder_path, image_file)

        # Open the image
        image = Image.open(image_path)

        # Apply transformations to the image
        transformed_image = Process()(image)
        transformed_image = transforms.ToTensor()(transformed_image)
        transformed_image = transforms.Normalize((0.5,), (0.5,))(transformed_image)

        # Save the transformed image in the output class folder
        output_image_path = os.path.join(output_class_folder, image_file)
        transforms.ToPILImage()(transformed_image).save(output_image_path)

        print("Processed image saved to:", output_image_path)


print("Image processing completed!")