from image_utils import load_image , edge_detection

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from image_utils import load_image, edge_detection  # Import functions from image_utils.py
from skimage.filters import median
from skimage.morphology import ball  # Used for noise suppression with ball structuring element

file_path = '/content/IMG_5288.jpeg' 
image_array = load_image(file_path)

# Perform edge detection
edge_image = edge_detection(image_array)

# Apply median filter to reduce noise using ball(3) as structuring element
clean_image = median(edge_image, ball(3))  # You can experiment with ball's radius (3 in this case)
    
# Convert the edge image to binary 
threshold = 150  #based on the histogram
edge_binary = clean_image > threshold  # (True for edges, False otherwise)

# Save the binary edge-detected image as a .png file
edge_image = Image.fromarray(edge_binary)
edge_image.save('my_edges.png')
