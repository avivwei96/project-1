import cv2
import os
import numpy as np
import warnings
from tqdm import tqdm


def extract_frames(video_path, frame_skip=1):
    # Check if the video file exists
    if not os.path.exists(video_path):
        print("Error: Video file not found")
        return
    i = 0
    # Open the video file
    video_capture = cv2.VideoCapture(video_path)

    frames = []

    # Read until video is completed
    while video_capture.isOpened():
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        if ret:
            # Append the frame to the list
            if i % frame_skip == 0:
                frames.append(frame)
            i += 1
        else:
            break

    # Release the video capture object
    video_capture.release()

    return frames


def cropLandScapeAndRotate(frame, height_top=0.75, height_bot=1, width_l=0.2, width_r=0.8):
    # Crop the frame to keep only the relevent part
    height, width = frame.shape[:2]
    cropped_frame = frame[int(height * height_top):int(height * height_bot) , int(width * width_l):int(width * width_r)].copy()
    frame[int(height * height_top):int(height * height_bot) , int(width * width_l):int(width * width_r)] = [0, 0, 0]
    
    # Rotate the cropped frame by 180 degrees
    rotated_frame = cv2.rotate(cropped_frame, cv2.ROTATE_90_CLOCKWISE)

    return rotated_frame


def paste_the_frame(frame, processed_part,
                    height_top=0.75, height_bot=1,
                    width_l=0.2, width_r=0.8):
    height, width = frame.shape[:2]
    frame[int(height * height_top):int(height * height_bot) , int(width * width_l):int(width * width_r)] = processed_part


def find_edges(frame , threshold_value=130, top_threshold_value=220, top_corner_size=90, bottom_corner_size=130, blurred_kernel_size=5, erode_kernel_size=3):
    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to convert the image to black and white
    _, black_and_white_frame = cv2.threshold(gray_frame, threshold_value, top_threshold_value, cv2.THRESH_BINARY)
    
    # Apply Gaussian blur to reduce noise in the binary image
    blurred_frame = cv2.GaussianBlur(black_and_white_frame, (blurred_kernel_size, blurred_kernel_size), 0)

    # Define a kernel for the dilation. You can adjust the size as needed.
    kernel = np.ones((erode_kernel_size, erode_kernel_size), np.uint8)

    # Apply dilation to the blurred binary image
    erode_frame = cv2.erode(blurred_frame, kernel, iterations=1)

    res_frame = crop_right_corners(erode_frame, top_corner_size=top_corner_size, bottom_corner_size=bottom_corner_size)

    # Apply Canny edge detection on the dilated black and white image
    edges = cv2.Canny(res_frame, 80, 150)

    return edges


def crop_right_corners(image, top_corner_size=50, bottom_corner_size=100):
    # Create a mask filled with ones (white) with the same dimensions as the image
    mask = np.ones_like(image) * 255

    # Get the dimensions of the image
    height, width = image.shape[:2]

    # Draw a black filled polygon for the top right corner
    top_triangle = np.array([
        [width - top_corner_size, 0],  # Starting point
        [width, 0],  # Top right corner of the image
        [width, top_corner_size + 200]  # End point of the diagonal
    ])
    cv2.fillPoly(mask, [top_triangle], (0, 0, 0))

    # Draw a black filled polygon for the bottom right corner
    bottom_triangle = np.array([
        [width - bottom_corner_size, height],  # Starting point
        [width, height],  # Bottom right corner of the image
        [width, height - bottom_corner_size - 200]  # End point of the diagonal
    ])
    cv2.fillPoly(mask, [bottom_triangle], (0, 0, 0))

    # Apply the mask to the image using bitwise_and to keep the central and left parts unchanged
    cropped_image = cv2.bitwise_and(image, mask)

    return cropped_image


def find_lines_with_hough_transform(edges, frame, mid_range_ret=25, mid_ret = 0.5, hough_threshold=15, minLineLength=5, maxLineGap=25):
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=hough_threshold, minLineLength=minLineLength, maxLineGap=maxLineGap)
    line_left_x, line_left_y, line_right_x, line_right_y, line_mid_x, line_mid_y = [], [], [], [], [], []
    mid_range = frame.shape[1] * (mid_range_ret/100)  # Define a range around the middle mid_range_ret% of the frame width

    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1 + 1e-8)  # Calculate the slope

            # Right lines
            if y1 > frame.shape[0] * mid_ret + mid_range and -2.9 < slope < 0:
                line_right_x.extend([x1, x2])
                line_right_y.extend([y1, y2])
            # Left lines
            elif y1 < frame.shape[0] * mid_ret - mid_range and 0 < slope < 2.9:
                line_left_x.extend([x1, x2])
                line_left_y.extend([y1, y2])
            # Midline
            elif np.abs(slope) < 1:
                line_mid_x.extend([x1, x2])
                line_mid_y.extend([y1, y2])

    return np.array(line_left_x), np.array(line_left_y), np.array(line_right_x), np.array(line_right_y), np.array(line_mid_x), np.array(line_mid_y)


def ransac_line_fit(x, y, threshold=10, num_iterations=1000, min_inliers=5):
    best_line = None
    best_inliers_indices = []

    if len(x) < 2 or len(y) < 2:
        return None, None  # Handle the case where there are not enough points
    
    for _ in range(num_iterations):
        # Randomly sample two points
        sample_indices = np.random.choice(len(x), 2, replace=False)
        sample_x = x[sample_indices]
        sample_y = y[sample_indices]
        
        try:
            # Fit a line to the sampled points
            warnings.filterwarnings('ignore')  # Ignore polyfit warnings
            line_params = np.polyfit(sample_x, sample_y, 1)
            line_function = np.poly1d(line_params)
            
            # Calculate distances of all points to the line
            distances = np.abs(line_function(x) - y)
            
            # Find inliers (points closer than threshold)
            inliers_indices = np.where(distances < threshold)[0]
            if len(inliers_indices) >= min_inliers and len(inliers_indices) > len(best_inliers_indices):
                best_inliers_indices = inliers_indices
                best_line = line_params
                
        except np.linalg.LinAlgError:
            # Handle the case where SVD did not converge
            continue  # Skip this iteration and try a new sample
    
    return best_line, best_inliers_indices


def draw_line_on_frame(frame, line_params, color=(0, 255, 0), thickness=2):
    # Unpack line parameters (slope and intercept)
    if line_params is None:
        return  # If line_params is None, there's nothing to draw
    slope, intercept = line_params

    # Ensure slope is not zero to avoid division by zero error
    if np.any(slope == 0):
        slope += 0.0000001

    # Set x1 to zero and calculate corresponding y1 using the intercept
    x1 = 0
    y1 = int(intercept)

    # Keep y2 at the bottom of the frame and calculate corresponding x2
    x2 = frame.shape[1]
    y2 = int(slope * x2 + intercept)

    # Draw the line on the frame
    cv2.line(frame, (x1, y1), (x2, y2), color, thickness)


def combine_lines(old_line, n_line, a=0.95):
    if np.any(old_line == None):
        return n_line
    if np.any(n_line == None):
        return old_line

    # To check if the line change is not resable
    if n_line[0] * old_line[0] < 0:
        return old_line

    res = np.array([a * old_line[0] + (1 - a) * n_line[0], a * old_line[1] + (1 - a) * n_line[1]])
    return res


def process_frame(frame):
    # Rotate and crop the frame
    Rotated_Cropped_frame = cropLandScapeAndRotate(frame)

    # Apply edge detection to the rotated and cropped frame
    canny_frame = find_edges(Rotated_Cropped_frame)

    # Use Hough Transform to find left, right, and midline lines in the edge-detected image
    line_left_x, line_left_y, line_right_x, line_right_y, line_mid_x, line_mid_y = find_lines_with_hough_transform(canny_frame, Rotated_Cropped_frame)

    # Return the processed frame along with the coordinates of the left, right, and midline lines
    return Rotated_Cropped_frame, line_left_x, line_left_y, line_right_x, line_right_y, line_mid_x, line_mid_y


def update_line(curr_line, new_line_data):
    new_line = ransac_line_fit(new_line_data[0], new_line_data[1])
    updated_line = combine_lines(curr_line, new_line[0])
    return updated_line, new_line


def draw_line_if_exists(frame, line, color, thickness):
    if line is not None:
        draw_line_on_frame(frame, line, color=color, thickness=thickness)


def process_video_frames(frames, max_mid_time=40):
    """
    Processes each frame from the video, updating and drawing lane lines, midline, and the mid-range area for debugging.
    Parameters:
    - frames: A list of video frames to process.
    - max_mid_time: Maximum number of consecutive frames the midline is drawn before being reset.
    - mid_range_ret: Percentage of the frame width considered as the mid-range area.
    Returns:
    - A list of processed frames with debugging visualizations.
    """
    n_frames = []  # Initialize the list to store processed frames
    curr_left_line, curr_right_line, curr_mid_line = None, None, None  # Initialize current lane lines and midline to None
    mid_counter = 1


    # Loop through each frame in the video
    for frame in tqdm(frames):
        # Rotate and crop the frame, then apply edge detection and find line segments using Hough Transform
        processed_frame, line_left_x, line_left_y, line_right_x, line_right_y, line_mid_x, line_mid_y = process_frame(frame)

        # Update and draw lines based on detections
        curr_left_line, _ = update_line(curr_left_line, (line_left_x, line_left_y))
        curr_right_line, _ = update_line(curr_right_line, (line_right_x, line_right_y))
        curr_mid_line, n_mid = update_line(curr_mid_line, (line_mid_x, line_mid_y))

        if mid_counter != 1 or curr_mid_line is not None:
            # Draw midline
            if mid_counter < 2:
                curr_left_line = curr_right_line = None

            if mid_counter % max_mid_time == 0:
                curr_mid_line = None
                mid_counter = 0
            mid_counter += 1
            if n_mid is None:
                curr_mid_line = curr_left_line
        else:
            # Draw left and right lines
            draw_line_if_exists(processed_frame, curr_right_line, color=(0, 255, 0), thickness=2)  # Green for right line
            draw_line_if_exists(processed_frame, curr_left_line, color=(0, 0, 255), thickness=2)  # Blue for left line

        # Rotate the processed frame back to its original orientation and add it to the list of processed frames
        # n_frames.append(cv2.rotate(processed_frame, cv2.ROTATE_90_COUNTERCLOCKWISE))

        paste_the_frame(frame, cv2.rotate(processed_frame, cv2.ROTATE_90_COUNTERCLOCKWISE))
        if mid_counter != 1:
            frame = cv2.putText(frame, "Lane Switch Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        n_frames.append(frame)


    return n_frames  # Return the list of processed frames


def display_frames(n_frames, delay=25):
    """
    Displays each frame in the list of processed frames.
    Parameters:
    - n_frames: A list of processed frames to display.
    The function displays each frame in a window and waits for a key press
    to proceed to the next frame. Pressing 'q' will exit the loop.
    """
    for frame in n_frames:
        cv2.imshow('Frame', frame)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()




def save_frames_as_video(frames, output_path, fps):
    # Determine the width and height of frames
    height, width, _ = frames[0].shape

    # Initialize VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Write frames to video
    for frame in frames:
        out.write(frame)

    # Release VideoWriter
    out.release()
    print(f"Video saved as {output_path}")


def main():
    video_path = "night/data/nightRoadCam.mp4"
    frames = extract_frames(video_path, frame_skip=1)
    n_frames = process_video_frames(frames)
    save_frames_as_video(n_frames, "data/night_project_res.mp4", fps=25)

if __name__ == "__main__":
    main()