import cv2
import numpy as np
from sklearn.linear_model import RANSACRegressor
import os
from tqdm import tqdm
from scipy.optimize import minimize
from scipy.interpolate import interp1d


#---------------AKAZE----------------

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

    return keypoints, descriptors




def match_features(descriptors1, descriptors2):
    # Define FLANN parameters for ORB
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm = FLANN_INDEX_LSH,
                        table_number = 6,
                        key_size = 12,
                        multi_probe_level = 1) 
    search_params = dict(checks=50)

    # Initialize FLANN matcher    match_map = {}  # Initialize a dictionary to hold the match_map for this video
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Match descriptors
    raw_matches = flann.knnMatch(np.asarray(descriptors1, np.uint8), np.asarray(descriptors2, np.uint8), k=2)

    # Apply ratio test
    good_matches = []
    for matches in raw_matches:
        if len(matches) == 2:
            m, n = matches
            if m.distance < 0.6 * n.distance:
                good_matches.append(m)

    return good_matches
# #------------------------------------


def construct_trajectories(matches, keypoints1, keypoints2, existing_trajectories=None, match_map=None, threshold=50):
    # Initialize dictionaries to hold the trajectories and match_map
    trajectories = existing_trajectories if existing_trajectories is not None else {}
    match_map = match_map if match_map is not None else {}

    # Iterate over all matches
    for match in matches:
        # Get the keypoints
        kp1 = keypoints1[match.queryIdx].pt
        kp2 = keypoints2[match.trainIdx].pt

        # If the keypoint index is in match_map, it was matched in the previous frame
        if match.queryIdx in match_map:
            # Get the id of the keypoint in the trajectories
            traj_id = match_map[match.queryIdx]

            # Calculate the distance between the last point in the trajectory and the new point
            distance = np.linalg.norm(np.array(trajectories[traj_id][-1]) - np.array(kp2))

            # If the distance is less than the threshold, append the new keypoint to the corresponding trajectory
            if distance <= threshold:
                trajectories[traj_id].append(kp2)
                # Update the match_map to point to the new keypoint
                match_map[match.trainIdx] = traj_id
        # Otherwise, start a new trajectory with the two keypoints
        else:
            # Create a new id for the trajectory
            traj_id = len(trajectories)
            # Add the trajectory to the trajectories dictionary
            trajectories[traj_id] = [kp1, kp2]
            # Add the new keypoint to the match_map
            match_map[match.trainIdx] = traj_id

    return trajectories, match_map


def filter_trajectories(trajectories, F=None, min_length=3, variance_threshold=5, score_threshold=0.32):    #len 28 vt 500.000 for ORB. len 15 vt 50.000 for SIFT. AKAZE min_length=20, variance_threshold=50000, score_threshold=0.32. Prøver score threshold 10
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
        print(f"Failed to compute fundamental matrix for trajectories:")
        print(f"Trajectory 1 ({len(trajectory1_unique)} points): {trajectory1_unique}")
        print(f"Trajectory 2 ({len(trajectory2_unique)} points): {trajectory2_unique}")
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

def check_overlap(trajectory1, trajectory2):

    # Check if the trajectories overlap in time
    time_overlap = len(trajectory1) == len(trajectory2)

    # Check if the trajectories overlap in space
    # Calculate the distance between each pair of corresponding points in the two trajectories
    distances = np.linalg.norm(np.array(trajectory1) - np.array(trajectory2), axis=1)
    # Check if the maximum distance is less than a threshold (you may need to adjust this value)
    space_overlap = np.max(distances) < 50

    # The trajectories overlap in time and space if both checks pass
    overlap = time_overlap and space_overlap

    return overlap


def match_trajectories(trajectories1, trajectories2, check_overlap_flag=False):
    # Initialize a list to hold the matched trajectories
    matched_trajectories = []

    # Initialize a counter for rejected trajectories
    rejected_count = 0

    # Get the total number of iterations for tqdm
    total_iterations = len(trajectories1) * len(trajectories2)

    # Initialize the tqdm progress bar
    pbar = tqdm(total=total_iterations, desc="Matching trajectories")

    # Iterate over all pairs of trajectories
    for trajectory1 in trajectories1:
        for trajectory2 in trajectories2:
            # If check_overlap_flag is True, check if the trajectories overlap in time and space
            if not check_overlap_flag or check_overlap(trajectory1, trajectory2):
                # Find the optimal offset that minimizes the residual error
                offset = find_optimal_offset(trajectory1, trajectory2)

                # Calculate the residual error for this pair of trajectories with the optimal offset
                error = calculate_residual_error([point + offset for point in trajectory1], trajectory2)

                # Append the pair, their error, and the optimal offset to the list of matched trajectories
                matched_trajectories.append((trajectory1, trajectory2, error, offset))
            else:
                # If the trajectories do not overlap and check_overlap_flag is True, increment the rejected counter
                rejected_count += 1

            # Update the tqdm progress bar
            pbar.update()

    # Close the tqdm progress bar
    pbar.close()

    # Print the number of rejected trajectories
    print(f"Number of rejected trajectories due to no overlap: {rejected_count}")

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

def resize_frame(frame, desired_width=None, desired_height=1500):
    # Get the original frame size
    original_height, original_width = frame.shape[:2]

    # If both dimensions are provided, raise an error
    if desired_width is not None and desired_height is not None:
        raise ValueError("Only one of desired_width or desired_height can be provided.")

    # If neither dimension is provided, raise an error
    if desired_width is None and desired_height is None:
        raise ValueError("At least one of desired_width or desired_height must be provided.")

    # Calculate the ratio and the new dimensions
    if desired_width is not None:
        # Calculate the ratio of the new width to the old width
        ratio = desired_width / float(original_width)
        # Calculate the new height based on the ratio
        new_height = int(original_height * ratio)
        new_width = desired_width
    else:
        # Calculate the ratio of the new height to the old height
        ratio = desired_height / float(original_height)
        # Calculate the new width based on the ratio
        new_width = int(original_width * ratio)
        new_height = desired_height

    # Resize the frame
    resized_frame = cv2.resize(frame, (new_width, new_height))

    return resized_frame





def main():

    # Initialize the background subtractor
    backSub = cv2.createBackgroundSubtractorMOG2(history=15, varThreshold=40, detectShadows=False)


    # File paths
    BASE_DIR = "./sample_alborz_x/younes"
    # Load your videos
    CAPTURE_FILES = [
        os.path.join(BASE_DIR, "y1.MOV"),
        os.path.join(BASE_DIR, "y2.MOV"),
    ]
    total_frames = [get_total_frames(path) for path in CAPTURE_FILES]

    # Load the videos
    videos = [load_video(path) for path in CAPTURE_FILES]

    # Initialize lists to hold trajectories for each video
    trajectories_list = []
    match_map_list = []  # To hold the match_map for each video

    # Extract features and construct trajectories for each video
    first_frames = []
    first_frames_keypoints = []  # To hold the keypoints of the first frame of each video
    first_frames_descriptors = []  # To hold the descriptors of the first frame of each video

    num_training_frames = 50  # Number of frames to use for training the background subtractor

    for i, (video, frames) in enumerate(zip(videos, total_frames)):

        # Train the background subtractor
        for _ in range(num_training_frames):
            try:
                frame = next(video)
                backSub.apply(frame)
            except StopIteration:
                break  # if no more frames are available, stop the loop

        try:
            old_frame = next(video)
        except StopIteration:
            print(f"Video {i} has fewer than {num_training_frames + 1} frames.")
            continue
        first_frames.append(old_frame)
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

        # # Define ROI ifor horizontal vid
        # height, width = old_frame.shape[:2]
        # left_percent = 0.15  # Adjust this to change the left coverage
        # roi_height = int(height)
        # roi_start_x = int(width * left_percent)
        # roi_width = width - roi_start_x
        # roi_start = (0, roi_start_x)
        # roi_size = (roi_height, roi_width)

        # Define ROI for vertical vid
        height, width = old_frame.shape[:2]
        top_percent = 0.15  # Adjust this to change the top coverage
        roi_height = int(height * (1 - top_percent))  # Adjust the height of the ROI
        roi_start_y = int(height * top_percent)  # Start the ROI after the top 15%
        roi_width = width  # Keep the width the same
        roi_start = (roi_start_y, 0)  # Start point of the ROI
        roi_size = (roi_height, roi_width)  # Size of the ROI
       

        p0, desc0 = extract_features_from_frame(old_frame, roi_start, roi_size)
        first_frames_keypoints.append(p0)
        first_frames_descriptors.append(desc0)

        # Print the number of features
        print(f"Number of features in video {i}: {len(p0)}")

        # Initialize a dictionary to hold the trajectories for this video
        trajectories = {}
        match_map = {}  # Initialize a dictionary to hold the match_map for this video

        # Skip the frames used for training the background subtractor
        for _ in range(num_training_frames):
            next(video, None)

        for frame in tqdm(video, total=frames - num_training_frames):





            p1, desc1 = extract_features_from_frame(frame, roi_start, roi_size)

            # #---------- Visualize the features ----------
            # frame_with_keypoints = cv2.drawKeypoints(frame, p1, None, color=(0,255,0), flags=0)
            # # Draw rectangle around the ROI
            # cv2.rectangle(frame_with_keypoints, (roi_start[1], roi_start[0]), (roi_start[1]+roi_size[1], roi_start[0]+roi_size[0]), (0,0,255), 2)
            # # Resize the frame
            # resized_frame = resize_frame(frame_with_keypoints)
            # cv2.imshow(f'Features in video {i}', resized_frame)
            # cv2.waitKey(1)  # Display the image for 1 ms


            # Match features
            matches = match_features(desc0, desc1)


            # # ----------Visualize the matches -----------
            # img_matches = cv2.drawMatches(old_frame, p0, frame, p1, matches, None)
            # resized_frame_matches = resize_frame(img_matches)
            # cv2.imshow('Matches', resized_frame_matches)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
    

            # Select good points
            good_matches = matches

            # If we have keypoints for at least two frames, we can start constructing trajectories
            if len(good_matches) > 1:
                # Construct trajectories from the old and new points
                trajectories, match_map = construct_trajectories(good_matches, p0, p1, trajectories, match_map)



            # Now update the previous frame and previous points
            old_frame = frame
            p0 = p1
            desc0 = desc1

                        
            # -------------Troubleshoot the raw trajectories---------

            # print("Raw Example Trajectories:")
            # for traj_id, traj in list(trajectories.items())[:5]:  # Print the first 5 trajectories
            #     print(f'Trajectory {traj_id}: {traj}')

            # print("Lengths of Raw Example Trajectories:")
            # for traj_id, traj in list(trajectories.items())[:20]:  # Print the lengths of the first 5 trajectories
            #     print(f'Length of Trajectory {traj_id}: {len(traj)}')



            # --------------Visualize the trajectories---------
            # frame_copy = frame.copy()

            # for traj in trajectories.values():
            #     viz_trajectory = [(int(x), int(y)) for x, y in traj]
            #     for traj_i in range(1, len(viz_trajectory)):
            #         cv2.line(frame_copy, viz_trajectory[traj_i-1], viz_trajectory[traj_i], (0, 255, 0), 2)

            # cv2.imshow('Trajectories', frame_copy)
            # cv2.waitKey(1)  # Display the image for 1 ms



        match_map_list.append(match_map)




        # Print the number of trajectories
        print(f"Number of trajectories in video {i}: {len(trajectories)}")
        
        # If this is not the first video, calculate the fundamental matrix and filter trajectories
        if i > 0:
            F, fund_matches, p1, p2 = compute_fundamental_matrix(first_frames_keypoints[0], first_frames_descriptors[0], first_frames_keypoints[i], first_frames_descriptors[i])
            # Draw the matches
            
            print(len(p1), len(p2))
            print(max(m.queryIdx for m in matches), max(m.trainIdx for m in matches))

            img_matches = cv2.drawMatches(first_frames[0], p1, first_frames[i], p2, fund_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
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




        # Convert trajectory points to integers
        filtered_trajectories = [[(int(x), int(y)) for x, y in trajectory] for trajectory in filtered_trajectories]





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




# remove trajectories that change too much over the course of one frame? 

# Length of trajectory: 8, Variance: [23.83729528 23.60425794]
# Length of trajectory: 951, Variance: [1420.20947603  387.97274691]
# Length of trajectory: 27, Variance