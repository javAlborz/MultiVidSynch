
import cv2
import numpy as np
from sklearn.linear_model import RANSACRegressor
import os
from tqdm import tqdm
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt




#--------------AKAZE-----------------
def extract_features_from_frame(frame, roi_start, roi_size):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Define ROI in the frame
    roi = gray[roi_start[0]:roi_start[0]+roi_size[0], roi_start[1]:roi_start[1]+roi_size[1]]

    # Detect features using AKAZE in the ROI
    akaze = cv2.AKAZE_create(descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB,
                             descriptor_size=0,
                             descriptor_channels=3,
                             threshold=0.001,
                             nOctaves=4,
                             nOctaveLayers=4)
    keypoints, descriptors = akaze.detectAndCompute(roi, None)

    # Adjust keypoint coordinates to be relative to the entire frame
    for keypoint in keypoints:
        keypoint.pt = (keypoint.pt[0] + roi_start[1], keypoint.pt[1] + roi_start[0])

    # Draw keypoints on the frame for visualization
    img_keypoints = cv2.drawKeypoints(frame, keypoints, None, color=(0, 255, 0))  # Draw keypoints in green color

    return keypoints, descriptors, img_keypoints




def match_features(descriptors1, descriptors2):
    # Define FLANN parameters for ORB
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm = FLANN_INDEX_LSH,
                        table_number = 6,
                        key_size = 12,
                        multi_probe_level = 1) 
    search_params = dict(checks=50)

    # Initialize FLANN matcher
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Match descriptors
    raw_matches = flann.knnMatch(np.asarray(descriptors1, np.uint8), np.asarray(descriptors2, np.uint8), k=2)

    # Apply ratio test
    good_matches = []
    for matches in raw_matches:
        if len(matches) == 2:
            m, n = matches
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

    return good_matches

def resize_frame(frame, desired_width=2500):
    # Get the original frame size
    original_height, original_width = frame.shape[:2]

    # Calculate the ratio of the new width to the old width
    ratio = desired_width / float(original_width)

    # Calculate the new height based on the ratio
    desired_height = int(original_height * ratio)

    # Resize the frame
    resized_frame = cv2.resize(frame, (desired_width, desired_height))

    return resized_frame


def match_first_frames():
    # Define ROI parameters
    left_percent = 0.15  # Adjust this to change the left coverage
    
    # Define the base directory and video file paths
    BASE_DIR = "./sample_alborz_x/younes"
    video_paths = [
        os.path.join(BASE_DIR, "y1_v2.mov"),
        os.path.join(BASE_DIR, "y2_v2.mov"),
    ]
    
    # Initialize lists to hold keypoints and descriptors of the first frames
    first_frame_keypoints = []
    first_frame_descriptors = []
    first_frames = []
    first_frames_keypoints_images = []  # Add this line
    
    # Extract features from the first frame of each video
    for video_path in video_paths:
        video = cv2.VideoCapture(video_path)
        ret, frame = video.read()
        if ret:
            height, width = frame.shape[:2]
            roi_height = int(height)
            roi_start_x = int(width * left_percent)
            roi_width = width - roi_start_x
            roi_start = (0, roi_start_x)
            roi_size = (roi_height, roi_width)
            keypoints, descriptors, img_keypoints = extract_features_from_frame(frame, roi_start, roi_size)  # Modify this line
            first_frame_keypoints.append(keypoints)
            first_frame_descriptors.append(descriptors)
            first_frames.append(frame)
            first_frames_keypoints_images.append(img_keypoints)  # Add this line
        video.release()
    
    # Match features between the first frames
    matches = match_features(first_frame_descriptors[0], first_frame_descriptors[1])
    print("Number of good matches:", len(matches))
    # Compute the fundamental matrix
    points1 = np.float32([first_frame_keypoints[0][m.queryIdx].pt for m in matches])
    points2 = np.float32([first_frame_keypoints[1][m.trainIdx].pt for m in matches])
    F, _ = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC)
    
    # Draw the matches
    img_matches = cv2.drawMatches(first_frames[0], first_frame_keypoints[0], first_frames[1], first_frame_keypoints[1], matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # # Display the image with matches using OpenCV
    # resized_frame = resize_frame(img_matches)

    # cv2.imshow("Matched Features", resized_frame)
    # cv2.waitKey(0)  # Wait indefinitely until a key is pressed
    # cv2.imshow("Keypoints", img_keypoints)
    # cv2.destroyAllWindows()  # Close the window

    # # At the end of the function, display the images with keypoints
    # for i, img in enumerate(first_frames_keypoints_images):
    #     resized_frame = resize_frame(img)
    #     cv2.imshow(f"Video {i+1} - Keypoints", resized_frame)

    # cv2.waitKey(0)  # Wait indefinitely until a key is pressed
    # cv2.destroyAllWindows()  # Close the window

    return F, matches

# Run the function
F, matches = match_first_frames()




#print number of matches
#Specialized ROI
#try different frames with only the boxers    

#space filter?

#It's okay to write about approach even if I did not get good results
