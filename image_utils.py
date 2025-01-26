import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from PIL import Image
from skimage.filters import median
from skimage.morphology import disk  # Use disk for 2D structuring element

def load_image(file_path):
    """Reads an image from the given file path and converts it into a NumPy array."""
    image = Image.open(file_path)  # Open the image using Pillow
    image_array = np.array(image)  # Convert the image into a NumPy array
    return image_array

# Test the function with an example image
file_path = '/content/IMG_5288.jpeg'  # Replace with your actual image path
image_array = load_image(file_path)

def edge_detection(image_array):
    """Perform edge detection using Sobel filters."""
    # Convert the image to grayscale by averaging over the color channels
    mean_image = np.mean(image_array, axis=2)  # Convert to grayscale by averaging RGB channels
    
    # Define Sobel filters for vertical (Y) and horizontal (X) edge detection
    kernelY = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])  # Vertical edge detection filter
    kernelX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # Horizontal edge detection filter
    
    # Apply the filters using convolution (zero padding by default)
    edgeY = convolve2d(mean_image, kernelY, mode='same', boundary='symm')  # Vertical edges
    edgeX = convolve2d(mean_image, kernelX, mode='same', boundary='symm')  # Horizontal edges
    
    # Calculate the magnitude of the gradient (edge strength)
    edgeMAG = np.sqrt(edgeX**2 + edgeY**2)
    
    return edgeMAG

# Perform edge detection
edge_image = edge_detection(image_array)
