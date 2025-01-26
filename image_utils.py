from image_utils import load_image, edge_detection
from PIL import Image
from skimage.filters import median
from skimage.morphology import ball
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

def edge_detection(image_array):
    mean_looki = image_array.mean(axis=2)
    filtered_looki_Y = convolve2d(mean_looki, np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]))
    filtered_looki_X = convolve2d(mean_looki, np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]))
    edgeMAG = np.sqrt(filtered_looki_X**2 + filtered_looki_Y**2)
    return edgeMAG

def load_image(file_path):
    image = Image.open(file_path)
    image_array = np.array(image)
    return image_array

file_path = '/content/looki.jpeg'
image_array = load_image(file_path)
print(image_array.shape)
