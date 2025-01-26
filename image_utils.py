import numpy as np
from scipy.ndimage import convolve2d
from PIL import Image
from skimage.filters import median
from skimage.morphology import ball

def load_image(file_path):
    image = Image.open(file_path) 
    image_array = np.array(image)  # Convert the image into a NumPy array
    return image_array

def edge_detection(image_array):
    mean_image = np.mean(image_array, axis=2)  # Convert to grayscale
    
    kernelY = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])  # Vertical edges
    kernelX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # Horizontal edges
    
    # Apply the filters to the grayscale image using convolution
    edgeY = convolve2d(mean_image, kernelY)  # Vertical edges
    edgeX = convolve2d(mean_image, kernelX)  # Horizontal edges
    
    # Calculate edgeMAG
    edgeMAG = np.sqrt(edgeX**2 + edgeY**2)
    
    return edgeMAG
