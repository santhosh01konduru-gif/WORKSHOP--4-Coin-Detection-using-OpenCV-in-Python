# WORKSHOP--4-Coin-Detection-using-O#penCV-in-Python
# NAME: konduru santhosh
# ROL.No: 212225240074
## Aim

To implement grayscale morphological operations using OpenCV for image preprocessing and edge-based detection, and to display the detected contours on the input image.

## Software Required

- Python
- Jupyter Notebook
- OpenCV (`cv2`)
- NumPy
- Matplotlib

## Algorithm

1. Import the required Python libraries.
2. Read the input image using OpenCV.
3. Convert the input image from BGR to grayscale.
4. Apply Gaussian Blur to reduce image noise.
5. Create a morphological kernel using NumPy.
6. Apply erosion to the preprocessed grayscale image.
7. Apply dilation to the eroded image.
8. Apply Canny Edge Detection to detect edges.
9. Find the external contours from the detected edges.
10. Draw the detected contours on the original image using green lines.
11. Convert the images from BGR to RGB for displaying with Matplotlib.
12. Display the original image and the processed image side by side.

## Program



## Develop by: V.Siri Sai

## Reg no : 212225240181





```


import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = "C:/Users/acer/OneDrive/Desktop/YOLOv4_Webcam/bird.jpg"

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred

def detect_fractures(preprocessed, original):
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(preprocessed, kernel, iterations=1)
    dilation = cv2.dilate(erosion, kernel, iterations=1)
    edges = cv2.Canny(dilation, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = original.copy()
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
    return result

def present_results(original_image, processed_image):
    original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    processed_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original_rgb)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Fracture Detected Image")
    plt.imshow(processed_rgb)
    plt.axis("off")

    plt.show()

image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found. Check the file path.")
else:
    preprocessed = preprocess_image(image)
    fracture_detected_image = detect_fractures(preprocessed, image)
    present_results(image, fracture_detected_image)

```

<img width="818" height="278" alt="image" src="https://github.com/user-attachments/assets/b0a7dd25-59a5-4b28-a593-d86c2d8ef86e" />


##  Result

The grayscale morphological operations, Gaussian filtering, Canny edge detection, and contour detection are successfully applied to the input image.
