from image_utils import load_image as li
from image_utils import edge_detection as ed

file_path = '/content/IMG_5288.jpeg'  # Replace with your actual image path
image_array = load_image(file_path)

# Perform edge detection on the image
edge_image = edge_detection(image_array)

# Apply median filter to reduce noise using the 2D slice of the 3D ball
# Use the ball(3) structuring element for 2D filtering
clean_image = median(edge_image, ball(2)[:, :, 0])  # Take only the 2D slice (first slice)

# Convert the edge image to binary using a threshold value
threshold = 150  # based on the histogram
edge_binary = clean_image > threshold  # binary image (True for edges, False otherwise)

# Save the binary edge-detected image as a .png file
edge_image = Image.fromarray((edge_binary * 255).astype(np.uint8))  # Convert boolean to 0-255
edge_image.save('my_edges.png')
