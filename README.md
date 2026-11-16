# Video Synchronization Based on Feature Trajectories
Full implementation of a video synchronization algorithm based on feature trajectories, as described in the paper [Feature-Based Multi-Video Synchronization with Subframe Accuracy, A. Elhayek et al](https://vcai.mpi-inf.mpg.de/files/DAGM2012/DAGM2012.pdf). The algorithm aims to synchronize multiple videos by extracting features, constructing trajectories, and matching these trajectories across videos.

## Introduction

Synchronizing videos is a crucial task in many multimedia applications ranging from film production to sports broadcasting. Traditional methods often rely on timestamps or manual alignment, which can be error-prone or time-consuming. This implementation offers an automated approach by leveraging traditional computer vision techniques. By extracting features from video frames and constructing trajectories, the algorithm can determine the synchronization parameters that align videos in time.

![Trajectory Construction Demonstration](./assets/trajectory_construction.gif)

## Features

- **Feature Extraction**: Uses AKAZE for robust and efficient feature extraction.
- **Trajectory Construction**: Constructs trajectories based on feature matches across consecutive frames.
- **Trajectory Filtering**: Filters out noisy and uninformative trajectories.
- **Matching Trajectories**: Matches trajectories between two videos to determine synchronization parameters.
- **Visualization**: Visualize keypoints, matches, and trajectories for debugging and understanding.

## User Guide

### Installation

```bash
git clone https://github.com/javAlborz/MultiVidSynch.git
cd MultiVidSynch
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Usage

1. Set `BASE_DIR` and `CAPTURE_FILES` in `feature_based_approach/OTP.py` to your video files.
2. Run from the algorithm's directory so its relative sample paths resolve:

   ```bash
   cd feature_based_approach
   python OTP.py
   ```

3. The script extracts features, constructs trajectories, and estimates temporal alignment between the videos. Results are printed to the console.
4. For visualization, uncomment the visualization sections in the script.

### Contribution

Contributions are welcome! Please submit a pull request or open an issue if you have improvements or find any bugs.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

