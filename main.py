import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from image_utils import load_image, edge_detection  # Import functions from image_utils.py
from skimage.filters import median
from skimage.morphology import ball  # Used for noise suppression with ball structuring element

# Step 1: Load the color image using the load_image function from image_utils
file_path = '/content/IMG_5288.jpeg'  # Update with the correct image path
image_array = load_image(file_path)  # Load image

# Step 2: Perform edge detection on the image
edge_image = edge_detection(image_array)  # Detect edges in the image

# Step 3: Apply median filter to reduce noise using ball(5) as structuring element
clean_image = median(edge_image, ball(5))  # Experiment with the radius (3 -> 5)

# Step 4: Convert the edge-detected image to binary using a threshold value
# Automatically determine a threshold based on the image's pixel range
threshold = np.mean(clean_image)  # Using the mean of the edge image as a threshold
edge_binary = clean_image > threshold  # Create binary image (True for edges, False otherwise)

# Step 5: Display the binary edge-detected image
plt.imshow(edge_binary, cmap='gray')  # Display image in grayscale
plt.axis('off')  # Turn off the axis for cleaner visualization
plt.show()  # Show the image

# Step 6: Save the binary edge-detected image as a .png file
edge_image = Image.fromarray((edge_binary * 255).astype(np.uint8))  # Convert boolean to 0-255
edge_image.save('my_edges.png')  # Save the result as 'my_edges.png'
