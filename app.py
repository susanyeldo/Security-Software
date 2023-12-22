import os
import streamlit as st
import cv2
import numpy as np
from datetime import datetime
import time
import matplotlib.pyplot as plt
import imutils
import re
import pytesseract
import shutil
import pandas as pd
import random
from PIL import Image





# -- Main function -- #
def main():
    st.title("Real-Time License Plate Detection")

    # Open the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Error: Unable to open the webcam.")
        return

    # Set video frame width and height
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Create a Streamlit placeholder to display the video stream
    video_placeholder = st.empty()

    # Create a button to capture a picture
    capture_button = st.button("Capture Picture", key = "manual")

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()

        if not ret:
            st.error("Error: Unable to read a frame from the webcam.")
            break

        # Display the frame in the Streamlit app
        video_placeholder.image(frame, channels="BGR", use_column_width=True)

        # Check for motion detection (you can customize this part)
        motion_detected = detect_motion(frame)

        if motion_detected or capture_button:
                      
            detect_license(frame)        
            
    # Release the webcam when the app is closed
    cap.release()




# -- Detect motion -- #
def detect_motion(frame):
    # Perform motion detection (you can customize this part)
    # For simplicity, let's use a basic method (e.g., frame differencing)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

    if 'background' not in globals():
        global background
        background = gray_frame.copy().astype("float")
        return False

    cv2.accumulateWeighted(gray_frame, background, 0.5)
    frame_delta = cv2.absdiff(gray_frame, cv2.convertScaleAbs(background))
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Count the number of non-zero pixels in the thresholded image
    motion_pixels = np.count_nonzero(thresh)

    # Customize the threshold based on your environment - number of pixel differences
    return motion_pixels > 300




# -- Detecting License Number from image -- #
def detect_license(frame):

    img = frame

    # Converting the image to grey scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Removing noise from image: Bilateral Filtering for smoothening the images while maintaining the edges
    bfilter = cv2.bilateralFilter(gray, 11, 11, 17)

    try:
        # Edge Detection using the Canny Algorithm
        edged = cv2.Canny(bfilter, 30, 200)

        # Finding the contours - licence plates have 4 keypoints
        keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(keypoints)

        # Sorting the top 10 contours
        contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]

        # Checking if the 4 key points chosen represent the number plate
        location = None
        for contour in contours:
            # approxPolyDP returns simlified resampled contour points
            approx = cv2.approxPolyDP(contour, 10, True)
            if len(approx) == 4:
                location = approx
                break

        # Creating the blank mask, contour, and then merging them
        mask = np.zeros(gray.shape, np.uint8)
        new_image = cv2.drawContours(mask, [location], 0, 255, -1)
        new_image = cv2.bitwise_and(img, img, mask = mask)

        # Converting the isolated licence plate to grey scale
        (x, y) = np.where(mask == 255)
        (x1, y1) = (np.min(x), np.min(y))
        (x2, y2) = (np.max(x), np.max(y))

        # Adding Buffer
        cropped_image = gray[x1:x2+3, y1:y2+3]
        # reading image
        extractedInformation = pytesseract.image_to_string(cropped_image)
        check_vehicle_number(extractedInformation, frame)

    except Exception as e:
        return
       



# -- Checking the Registered Vehicle Numbers -- #
def check_vehicle_number(extractedInformation, frame):
    extractedInformation = re.sub(r'[^a-zA-Z0-9]', '', extractedInformation)
    # Reading the csv file
    df = pd.read_csv('vehicle_numbers.csv')
    success_placeholder = st.empty()
    # Checking if the extracted information is in the csv file
    if extractedInformation in df.values:
        success_placeholder.success(f"Vehicle Number {extractedInformation} is Registered")
        # Save the captured image
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        captured_image_path = os.path.join("Images Captured", f"captured_image_{current_datetime}.jpg")
        cv2.imwrite(captured_image_path, frame)
        time.sleep(5)
        success_placeholder.empty()
    else:
        return
    





if __name__ == "__main__":
    main()
