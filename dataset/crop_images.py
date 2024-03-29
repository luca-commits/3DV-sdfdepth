from PIL import Image
import os
import argparse
import json

def center_crop_image(image_path, crop_pixels):
    with Image.open(image_path) as img:
        width, height = img.size

        # Compute new coordinates for the center crop
        left = crop_pixels
        top = crop_pixels
        right = width - crop_pixels
        bottom = height - crop_pixels
        
        # Check if the cropping is feasible
        if left < 0 or top < 0 or right > width or bottom > height:
            print(f"Cannot crop {crop_pixels}px from image {image_path} as its dimensions are ({width}x{height})")
            return

        cropped_img = img.crop((left, top, right, bottom))
        #create a new folder called cropped in the location of the image if it doesn't exist
        if not os.path.exists(os.path.join(os.path.dirname(image_path), "cropped")):
            os.makedirs(os.path.join(os.path.dirname(image_path), "cropped"))
        cropped_img.save(os.path.join(os.path.dirname(image_path), "cropped", os.path.basename(image_path)))

def center_crop_all_images(directory, crop_pixels):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                image_path = os.path.join(root, file)
                center_crop_image(image_path, crop_pixels)
            if file.lower().endswith(('.json')):
                params = json.load(open(os.path.join(root, file)))
                params["cx"] -= crop_pixels
                params["cy"] -= crop_pixels
                params["w"] -= 2*crop_pixels
                params["h"] -= 2*crop_pixels
                json.dump(params, open(os.path.join(root, "cropped", file), "w"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Crop given number of pixels on all sides of the image for all images in a folder')
    parser.add_argument('--basedir', type=str, required=True)
    parser.add_argument('--img_folder', type=str, required=True)
    parser.add_argument('--pixels', type=int, required=True)
    args = parser.parse_args()
    directory = os.path.join(args.basedir, args.img_folder)
    crop_pixels = 100  
    center_crop_all_images(directory, args.pixels)
