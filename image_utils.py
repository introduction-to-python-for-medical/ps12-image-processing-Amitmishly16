import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from PIL import Image
from skimage.filters import median
from skimage.morphology import disk  # Use disk for 2D structuring element

def load_image(file_path):
    """Load an image and convert it to a numpy array."""
    image = Image.open(file_path)
    image_array = np.array(image)
    return image_array

# Load the image
file_path = '/content/looki.jpeg'
image_array = load_image(file_path)
print(image_array.shape)

# Edge detection function
def edge_detection(image_array):
    mean_image = np.mean(image_array, axis=2)
    kernelY = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])  # Vertical edges
    kernelX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # Horizontal edges
    edgeY = convolve2d(mean_image, kernelY, mode='same', boundary='symm')
    edgeX = convolve2d(mean_image, kernelX, mode='same', boundary='symm')
    edgeMAG = np.sqrt(edgeX**2 + edgeY**2)
    edgeMAG = np.clip((edgeMAG - edgeMAG.min()) / (edgeMAG.max() - edgeMAG.min()) * 255, 0, 255)
    return edgeMAG
