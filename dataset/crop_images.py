from PIL import Image
import os
import argparse

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
        #create a new folder called cropped if it doesn't exist
        if not os.path.exists("cropped"):
            os.makedirs("cropped")
        cropped_img.save("cropped/" + os.path.basename(image_path))

def center_crop_all_images(directory, crop_pixels):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                image_path = os.path.join(root, file)
                center_crop_image(image_path, crop_pixels)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Crop given number of pixels on all sides of the image for all images in a folder')
    parser.add_argument('--basedir', type=str, required=True)
    parser.add_argument('--img_folder', type=str, required=True)
    parser.add_argument('--pixels', type=int, required=True)
    args = parser.parse_args()
    directory = os.path.join(args.basedir, args.img_folder)
    crop_pixels = 100  
    center_crop_all_images(directory, args.pixels)
