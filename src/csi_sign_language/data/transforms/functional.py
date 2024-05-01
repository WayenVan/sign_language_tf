import cv2
import numpy as np
from PIL import Image
def numpy2pil(video):
    return [Image.fromarray(frame) for frame in video]

def pil2numpy(video):
    return np.stack([np.array(frame) for frame in video], axis=0)

def rotate_and_crop(image, angle, crop_x, crop_y, crop_width, crop_height):
    # Read the image
    original_image = image
    # Get image dimensions
    height, width = original_image.shape[:2]
    # Calculate the center of the image
    center = (width // 2, height // 2)
    # Define the rotation matrix using the center of the image
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
    # Rotate the image
    rotated_image = cv2.warpAffine(original_image, rotation_matrix, (width, height))
    # Crop the rotated image
    cropped_image = rotated_image[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]
    return cropped_image

def adjust_bright(img, factor):
    #[h, w, rbg]
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # Scale the brightness channel
    hsv[:,:,2] = np.clip(hsv[:,:,2] * factor, 0, 255).astype(np.uint8)
    # Convert the image back to the BGR color space
    jittered_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return jittered_image

def to_gray(img):
    #[h, w, rbg]
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Duplicate the single channel to create a three-channel image
    gray_image_3_channels = cv2.merge([gray_image, gray_image, gray_image])
    return gray_image_3_channels

