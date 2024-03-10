import cv2
import numpy as np
import matplotlib.pyplot as plt

def convert_to_sketch(image_path, save_path_sketch):
    # Read the 3D image
    image = cv2.imread(image_path)
    
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to the grayscale image to reduce noise
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    # Detect edges in the image using Canny edge detection
    edges = cv2.Canny(blurred_image, threshold1=30, threshold2=100)
    
    # Invert the edges to get white lines on a black background
    inverted_edges = cv2.bitwise_not(edges)
    
    # Save the sketch image
    cv2.imwrite(save_path_sketch, inverted_edges)
    
    # Display the original image and the sketch using Matplotlib
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    axes[1].imshow(cv2.imread(save_path_sketch, cv2.IMREAD_GRAYSCALE), cmap='gray')
    axes[1].set_title('Sketch')
    axes[1].axis('off')
    plt.show()

# Provide the path to the 3D image
image_path = 'source_image.jpg'
save_path_sketch = 'sketch_image.jpg'
convert_to_sketch(image_path, save_path_sketch)
