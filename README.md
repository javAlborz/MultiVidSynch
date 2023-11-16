# Video Synchronization Based on Feature Trajectories

This repository contains an implementation of a video synchronization algorithm based on feature trajectories, as described in the paper [Feature-Based Multi-Video Synchronization with
Subframe Accuracy, A. Elhayek et al](https://vcai.mpi-inf.mpg.de/files/DAGM2012/DAGM2012.pdf). The algorithm aims to synchronize multiple videos by extracting features, constructing trajectories, and matching these trajectories across videos.

## Introduction

Synchronizing videos is a crucial task in many multimedia applications ranging from film production to sports broacasting. Traditional methods often rely on timestamps or manual alignment, which can be error-prone or time-consuming. This implementation offers an automated approach by leveraging tradiitional computer vision techniques. By extracting features from video frames and constructing trajectories, the algorithm can determine the synchronization parameters that align videos in time.

![Trajectory Construction Demonstration](./assets/trajectory_construction.gif)

## Features

- **Feature Extraction**: Uses AKAZE for robust and efficient feature extraction.
- **Trajectory Construction**: Constructs trajectories based on feature matches across consecutive frames.
- **Trajectory Filtering**: Filters out noisy and uninformative trajectories.
- **Matching Trajectories**: Matches trajectories between two videos to determine synchronization parameters.
- **Visualization**: Visualize keypoints, matches, and trajectories for debugging and understanding.

## User Guide

### Prerequisites
Install the required packages:

pip install -r requirements.txt


### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/video-synchronization.git
   cd video-synchronization

(Optional) Create a virtual environment:
    python3 -m venv venv
    source venv/bin/activate

### Usage

1. Place your videos in the `sample_alborz_x/younes` directory (or modify the `CAPTURE_FILES` variable in the script to point to your videos).

2. Run the main script:

3. The script will process the videos, extract features, construct trajectories, and determine synchronization parameters. The results will be printed to the console.

4. (Optional) For visualization, uncomment the visualization sections in the code to see keypoints, matches, and trajectories.

### Contribution

Contributions are welcome! Please submit a pull request or open an issue if you have improvements or find any bugs.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

