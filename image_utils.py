from image_utils import load_image, edge_detection
from PIL import Image
from skimage.filters import median
from skimage.morphology import ball
from matplotlib import pyplot as plt
from scipy.signal import convolve2d
import numpy as np

def load_image(file_path):
    image = Image.open(file_path)
    image_array = np.array(image)
    return image_array

file_path = '/content/looki.jpeg'
image_array = load_image(file_path)
print(image_array.shape)

def edge_detection(image_array):
    mean_image = np.mean(image_array, axis=2)
    kernelY = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    kernelX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    edgeY = convolve2d(mean_image, kernelY, mode='same', boundary='symm')
    edgeX = convolve2d(mean_image, kernelX, mode='same', boundary='symm')
    edgeMAG = np.sqrt(edgeX**2 + edgeY**2)
    return edgeMAG
