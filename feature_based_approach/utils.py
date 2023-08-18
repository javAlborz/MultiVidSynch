import cv2 

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

