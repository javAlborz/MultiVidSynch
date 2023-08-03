import cv2
import numpy as np
from sklearn.linear_model import RANSACRegressor
import os
from tqdm import tqdm
from scipy.optimize import minimize
from scipy.interpolate import interp1d


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

    return keypoints, descriptors



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
            if m.distance < 0.60 * n.distance:
                good_matches.append(m)

    return good_matches
# #------------------------------------  #originally 0.7. Try with 0.75 or 0.8


def construct_trajectories(matches, keypoints1, keypoints2, existing_trajectories=None):
    # Initialize a dictionary to hold the trajectories
    trajectories = existing_trajectories if existing_trajectories is not None else {}

    # Iterate over all matches
    for match in matches:
        # Get the keypoints
        kp1 = keypoints1[match.queryIdx].pt
        kp2 = keypoints2[match.trainIdx].pt

        # If the keypoint index is already in the dictionary, append the new keypoint to the trajectory
        if match.queryIdx in trajectories:
            trajectories[match.queryIdx].append(kp2)
        # Otherwise, start a new trajectory with the two keypoints
        else:
            trajectories[match.queryIdx] = [kp1, kp2]

    return trajectories


def filter_trajectories(trajectories, F=None, min_length=15, variance_threshold=50000, score_threshold=0.32):    #len 28 vt 500.000 for ORB. len 15 vt 50.000 for SIFT. AKAZE min_length=20, variance_threshold=50000, score_threshold=0.32
    # Initialize list to hold filtered trajectories
    filtered_trajectories = []

    # Iterate over all trajectories
    for i, trajectory in enumerate(trajectories):
        # Compute the variance of the spatial coordinates of the trajectory
        variance = np.var(trajectory, axis=0)

        # Check if the trajectory is long enough and not static
        if len(trajectory) >= min_length and np.any(variance > variance_threshold):
            
            # Print the length and variance of the trajectory
            print(f"Length of trajectory: {len(trajectory)}, Variance: {variance}")
            
            # If F is not None, compute the additional filtering criterion
            if F is not None:
                # Compute the tangents for each point in the trajectory
                tangents = np.diff(trajectory, axis=0)
                norms = np.linalg.norm(tangents, axis=1, keepdims=True)
                norms[norms == 0] = 1e-8  # to avoid division by zero
                tangents = tangents / norms

                # Compute the corresponding epipolar lines for each point in the trajectory
                epilines = cv2.computeCorrespondEpilines(np.array(trajectory[:-1]).reshape(-1, 1, 2), 2, F).reshape(-1, 3)
                
                # Compute the cosine of the angle between the tangent and the epipolar line for each point in the trajectory
                cosines = np.abs(np.sum(tangents * epilines[:, :2], axis=1) / np.linalg.norm(epilines[:, :2], axis=1))

                # Compute the score for the trajectory
                score = np.sum(1 - cosines)
                print(f"Score of trajectory {i}: {score}")
                # If the score passes the threshold, append the trajectory to the list of filtered trajectories
                if score >= score_threshold:
                    filtered_trajectories.append(trajectory)
            else:
                # If F is None, append the trajectory to the list of filtered trajectories without the additional filtering
                filtered_trajectories.append(trajectory)

    return filtered_trajectories



#------------------------------------------------------------------------------------

def calculate_residual_error(trajectory1, trajectory2):
    # Ensure the trajectories are numpy arrays
    trajectory1 = np.array(trajectory1)
    trajectory2 = np.array(trajectory2)

    # If the trajectories have different lengths, interpolate the shorter one
    if len(trajectory1) != len(trajectory2):
        if len(trajectory1) < len(trajectory2):
            t = np.linspace(0, 1, len(trajectory1))
            t_new = np.linspace(0, 1, len(trajectory2))
            trajectory1 = np.column_stack((np.interp(t_new, t, trajectory1[:, 0]), np.interp(t_new, t, trajectory1[:, 1])))
        else:
            t = np.linspace(0, 1, len(trajectory2))
            t_new = np.linspace(0, 1, len(trajectory1))
            trajectory2 = np.column_stack((np.interp(t_new, t, trajectory2[:, 0]), np.interp(t_new, t, trajectory2[:, 1])))

    # Get unique pairs from each trajectory
    trajectory1_unique = np.unique(trajectory1, axis=0)
    trajectory2_unique = np.unique(trajectory2, axis=0)

    # Trim the longer trajectory to the length of the shorter one
    if len(trajectory1_unique) < len(trajectory2_unique):
        trajectory2_unique = trajectory2_unique[:len(trajectory1_unique)]
    elif len(trajectory2_unique) < len(trajectory1_unique):
        trajectory1_unique = trajectory1_unique[:len(trajectory2_unique)]

    # Compute the fundamental matrix between the unique pairs
    F, _ = cv2.findFundamentalMat(trajectory1_unique, trajectory2_unique, cv2.FM_RANSAC, 1.0, 0.99, 5000)

    # Check if F is not None and its size
    if F is None or F.shape != (3, 3):
        print(f"Failed to compute fundamental matrix for trajectories:\n{trajectory1_unique}\n{trajectory2_unique}")
        return float('inf')

    # Compute the epilines for each point in trajectory1
    epilines1 = cv2.computeCorrespondEpilines(trajectory2_unique.reshape(-1, 1, 2), 2, F)

    # Compute the epilines for each point in trajectory2
    epilines2 = cv2.computeCorrespondEpilines(trajectory1_unique.reshape(-1, 1, 2), 1, F)

    # Compute the residual error for each pair of corresponding points
    error = 0
    for i in range(len(trajectory1_unique)):
        error += abs(trajectory1_unique[i][0]*epilines2[i][0][0] + trajectory1_unique[i][1]*epilines2[i][0][1] + epilines2[i][0][2])
        error += abs(trajectory2_unique[i][0]*epilines1[i][0][0] + trajectory2_unique[i][1]*epilines1[i][0][1] + epilines1[i][0][2])

    return error


def find_optimal_offset(trajectory1, trajectory2):
    # Define the cost function
    def cost_function(offset):
        # Shift trajectory2 by the offset
        shifted_trajectory2 = [point + offset for point in trajectory2]

        # Interpolate the shifted_trajectory2 to match the length of trajectory1
        f = interp1d(range(len(shifted_trajectory2)), shifted_trajectory2, axis=0, fill_value="extrapolate")
        interpolated_trajectory2 = f(np.linspace(0, len(shifted_trajectory2)-1, len(trajectory1)))

        # Compute the residual error between trajectory1 and the interpolated_trajectory2
        error = calculate_residual_error(trajectory1, interpolated_trajectory2)

        return error

    # Use scipy.optimize.minimize to find the offset that minimizes the cost function
    result = minimize(cost_function, 0, method='Powell')
    # Return the optimal offset
    return result.x


def match_trajectories(trajectories1, trajectories2):
    # Initialize a list to hold the matched trajectories
    matched_trajectories = []

    # Get the total number of iterations for tqdm
    total_iterations = len(trajectories1) * len(trajectories2)

    # Initialize the tqdm progress bar
    pbar = tqdm(total=total_iterations, desc="Matching trajectories")

    # Iterate over all pairs of trajectories
    for trajectory1 in trajectories1:
        for trajectory2 in trajectories2:
            # Find the optimal offset that minimizes the residual error
            offset = find_optimal_offset(trajectory1, trajectory2)

            # Calculate the residual error for this pair of trajectories with the optimal offset
            error = calculate_residual_error([point + offset for point in trajectory1], trajectory2)

            # Append the pair, their error, and the optimal offset to the list of matched trajectories
            matched_trajectories.append((trajectory1, trajectory2, error, offset))

            # Update the tqdm progress bar
            pbar.update()

    # Close the tqdm progress bar
    pbar.close()

    # Sort the matched trajectories by error in ascending order
    matched_trajectories.sort(key=lambda x: x[2])

    # Return the matched trajectories
    return matched_trajectories

#------------------------------------------------------------------------------------


# And finally, we modify the synchronize_videos function to use the optimal offsets
def synchronize_videos(matched_trajectories):
    # Prepare the input data and targets for the RANSAC algorithm
    X = []
    y = []
    for trajectory1, trajectory2, error, offset in matched_trajectories:
        # Use the individual spatial coordinates of each point in the trajectory as input
        X.extend(trajectory1)
        X.extend([point + offset for point in trajectory2])

        # Use the calculated residual error as the target
        y.extend([error] * len(trajectory1))
        y.extend([error] * len(trajectory2))

    # Convert X and y to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Reshape X to have shape (n_samples, n_features)
    X = X.reshape(X.shape[0], -1)

    # Initialize the RANSAC model
    ransac = RANSACRegressor()

    # Fit the RANSAC model
    ransac.fit(X, y)

    # Select the inliers among the trajectory pairs
    inliers = X[ransac.inlier_mask_]

    # Estimate the offsets based on the inliers
    offsets = np.mean(inliers, axis=0)

    return offsets


def compute_fundamental_matrix(p1, desc1, p2, desc2):
    # Match features between the frames
    matches = match_features(desc1, desc2)

    # Print the number of matches
    print(f"Number of matches for initial fundamental matrix: {len(matches)}")

    # Get the coordinates of the matched keypoints
    points1 = np.float32([p1[m.queryIdx].pt for m in matches])
    points2 = np.float32([p2[m.trainIdx].pt for m in matches])

    # Print the coordinates of the points
    print("Points1:", points1)
    print("Points2:", points2)

    # Compute the fundamental matrix
    F, _ = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC)

    # Print the shape of F
    print("Shape of F:", F.shape)

    return F, matches, p1, p2





def load_video(video_path):
    # Initialize a VideoCapture object
    video = cv2.VideoCapture(video_path)

    # Loop over all frames in the video
    while video.isOpened():
        # Read the next frame
        ret, frame = video.read()

        # If the frame was successfully read
        if ret:
            # Yield the frame
            yield frame
        else:
            # If no more frames are available, stop the loop
            break

    # Release the VideoCapture object
    video.release()

def get_total_frames(video_path):
    # Initialize a VideoCapture object
    video = cv2.VideoCapture(video_path)

    # Get the total number of frames
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Release the VideoCapture object
    video.release()

    return total_frames

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


def main():
    # File paths
    BASE_DIR = "./sample_alborz_x/00010_shi_qi_vs_ji_xuanhui_mtch_1"
    # Load your videos
    CAPTURE_FILES = [
        os.path.join(BASE_DIR, "beginning_00.mp4"),
        os.path.join(BASE_DIR, "beginning_01.mp4"),
        #os.path.join(BASE_DIR, "ending_02.mp4")
    ]
    total_frames = [get_total_frames(path) for path in CAPTURE_FILES]

    # Load the videos
    videos = [load_video(path) for path in CAPTURE_FILES]

    # Initialize lists to hold trajectories for each video
    trajectories_list = []

    # Extract features and construct trajectories for each video
    first_frames = []
    first_frames_keypoints = []  # To hold the keypoints of the first frame of each video
    first_frames_descriptors = []  # To hold the descriptors of the first frame of each video

    for i, (video, frames) in enumerate(zip(videos, total_frames)):
        old_frame = next(video)
        first_frames.append(old_frame)
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        # Define ROI in the main function
        height, width = old_frame.shape[:2]
        roi_percent = 0.8
        roi_size = (int(height*roi_percent), int(width*roi_percent))
        roi_start = (height//2 - roi_size[0]//2, width//2 - roi_size[1]//2)

        # Use the ROI in the feature extraction
        p0, desc0 = extract_features_from_frame(old_frame, roi_start, roi_size)
        first_frames_keypoints.append(p0)
        first_frames_descriptors.append(desc0)

        # Print the number of features
        print(f"Number of features in video {i}: {len(p0)}")

        # Initialize a dictionary to hold the trajectories for this video
        trajectories = {}

        for frame in tqdm(video, total=frames):

            p1, desc1 = extract_features_from_frame(frame, roi_start, roi_size)

            # #---------- Visualize the features ----------
            frame_with_keypoints = cv2.drawKeypoints(frame, p1, None, color=(0,255,0), flags=0)
            # Draw rectangle around the ROI
            cv2.rectangle(frame_with_keypoints, (roi_start[1], roi_start[0]), (roi_start[1]+roi_size[1], roi_start[0]+roi_size[0]), (0,0,255), 2)
            # Resize the frame
            resized_frame = resize_frame(frame_with_keypoints)
            cv2.imshow(f'Features in video {i}', resized_frame)
            cv2.waitKey(1)  # Display the image for 1 ms


            # Match features
            matches = match_features(desc0, desc1)

            # Select good points
            good_matches = [m for m in matches if m.distance < 0.75]

            # If we have keypoints for at least two frames, we can start constructing trajectories
            if len(good_matches) > 1:
                # Construct trajectories from the old and new points
                trajectories = construct_trajectories(good_matches, p0, p1, trajectories)

            # Now update the previous frame and previous points
            p0 = p1
            desc0 = desc1


        # Print the number of trajectories
        print(f"Number of trajectories in video {i}: {len(trajectories)}")

        # If this is not the first video, calculate the fundamental matrix and filter trajectories
        if i > 0:
            F, fund_matches, p1, p2 = compute_fundamental_matrix(first_frames_keypoints[0], first_frames_descriptors[0], first_frames_keypoints[i], first_frames_descriptors[i])
            # Draw the matches
            img_matches = cv2.drawMatches(first_frames[0], p1, first_frames[i], p2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            # Display the image with matches
            # Alternatively, write the image with matches to a file
            cv2.imwrite(f'matches_{i}.png', img_matches)

            # Check if F is valid and its size
            if F is None or F.shape != (3, 3):
                print(f"Failed to compute fundamental matrix between video 0 and video {i}.")
                continue  # Skip the rest of this iteration

            # Filter the trajectories using the calculated fundamental matrix
            filtered_trajectories = filter_trajectories(list(trajectories.values()), F)
                        
            # Print the number of filtered trajectories
            print(f"Number of filtered trajectories in video {i}: {len(filtered_trajectories)}")
        else:
            # For the first video, we don't have a pair to compute the fundamental matrix
            # So we just filter the trajectories based on length and variance
            filtered_trajectories = filter_trajectories(list(trajectories.values()), None)
            print(f"Number of filtered trajectories in video {i}: {len(filtered_trajectories)}")

        # Add the filtered trajectories for this video to the trajectories list
        trajectories_list.append(filtered_trajectories)


    # After processing all videos, re-filter the trajectories of the first video using the computed fundamental matrix
    if len(trajectories_list) > 1:
        F, fund_matches, p1, p2 = compute_fundamental_matrix(first_frames_keypoints[0], first_frames_descriptors[0], first_frames_keypoints[1], first_frames_descriptors[1])
        print("Fundamental matrix:", F)
        if F is not None and F.shape == (3, 3):
            print('Performing additional filtering on the first trajectory list')
            trajectories_list[0] = filter_trajectories(trajectories_list[0], F)
            print(f"Number of filtered trajectories in video {0}: {len(trajectories_list[0])}")
        else:
            print("Failed to compute fundamental matrix for re-filtering.")




    cv2.destroyAllWindows()  # Close all OpenCV windows


    # Initialize a dictionary to hold the synchronization parameters for each video
    sync_dict = {}

    # Match trajectories and synchronize videos for each pair of videos
    for i in range(1, len(trajectories_list)):
        # Match trajectories between the reference video and the current video
        matched_trajectories = match_trajectories(trajectories_list[0], trajectories_list[i])

        # Print the number of matched trajectories
        print(f"Number of matched trajectories between cam_00 and cam_0{i}: {len(matched_trajectories)}")

        # Synchronize the videos
        offsets = synchronize_videos(matched_trajectories)

        # Choose the first video as the reference video
        ref_offset = offsets[0]

        # Adjust the offsets of all the other videos relative to the reference video
        adjusted_offsets = [offset - ref_offset for offset in offsets]

        # Add the synchronization parameters for the current video to the dictionary
        sync_dict[f'cam_0{i}'] = int(adjusted_offsets[1])

    # Print the synchronization parameters
    print({'sync': sync_dict})


if __name__ == "__main__":
    main()


#wtf? reverted back one day



