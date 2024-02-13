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


def cropLandScapeAndRotate(frame):
    # Crop the frame to keep only the bottom three-fifths of the height and the middle three-fifths of the width
    height, width = frame.shape[:2]
    cropped_frame = frame[(height * 3) // 5:(height * 9) // 10 , width // 5:(width * 4) // 5]
    
    # Rotate the cropped frame by 180 degrees
    rotated_frame = cv2.rotate(cropped_frame, cv2.ROTATE_90_CLOCKWISE)
    
    return rotated_frame


def find_edges(frame, threshold_value=180):
    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to convert the image to black and white
    _, black_and_white_frame = cv2.threshold(gray_frame, threshold_value, 255, cv2.THRESH_BINARY)
    
    # Apply Gaussian blur to reduce noise in the binary image
    blurred_frame = cv2.GaussianBlur(black_and_white_frame, (5, 5), 0)

    # Define a kernel for the dilation. You can adjust the size as needed.
    kernel = np.ones((3, 3), np.uint8)

    # Apply dilation to the blurred binary image
    dilated_frame = cv2.erode(blurred_frame, kernel, iterations=1)

    dilated_frame = crop_right_corners(dilated_frame, corner_size=100)

    # Apply Canny edge detection on the dilated black and white image
    edges = cv2.Canny(dilated_frame, 50, 150)

    return edges


def crop_right_corners(image, corner_size=50):
    # Create a mask filled with ones (white) with the same dimensions as the image
    mask = np.ones_like(image) * 255
    
    # Get the dimensions of the image
    height, width = image.shape[:2]
    
    # Define the size of the corners to be cropped
    # 'corner_size' specifies the height and width of the corner rectangles
    
    # Fill the top right corner of the mask with black (0)
    mask[:corner_size, width - corner_size:] = 0
    
    # Fill the bottom right corner of the mask with black (0)
    mask[height - corner_size:, width - corner_size:] = 0

    # Apply the mask to the image using bitwise_and to keep the central and left parts unchanged
    cropped_image = cv2.bitwise_and(image, mask)

    return cropped_image


def find_lines_with_hough_transform(edges, frame):
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=20, minLineLength=25, maxLineGap=10)
    line_left_x, line_left_y, line_right_x, line_right_y = [], [], [], []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1 + 1e-6)
            
            if y1 > frame.shape[0]//2:
                if -1.8 < slope < 0:
                    line_right_x.extend([x1, x2])
                    line_right_y.extend([y1, y2])
            else:
                if 0 < slope < 1.4:
                    line_left_x.extend([x1, x2])
                    line_left_y.extend([y1, y2])

    return np.array(line_left_x), np.array(line_left_y), np.array(line_right_x), np.array(line_right_y)


def ransac_line_fit(x, y, threshold=10, num_iterations=1000, min_inliers=2):
    best_line = None
    best_inliers_indices = []

    if len(x) < 2 or len(y) < 2:
        return None, None  # Handle the case where there are not enough points
    
    for _ in range(num_iterations):
        # Randomly sample two points
        sample_indices = np.random.choice(len(x), 2, replace=False)
        sample_x = x[sample_indices]
        sample_y = y[sample_indices]
        
        # Fit a line to the sampled points
        warnings.filterwarnings('ignore')
        line_params = np.polyfit(sample_x, sample_y, 1)
        line_function = np.poly1d(line_params)
        
        # Calculate distances of all points to the line
        distances = np.abs(line_function(x) - y)
        
        # Find inliers (points closer than threshold)
        inliers_indices = np.where(distances < threshold)[0]
        if len(inliers_indices) >= min_inliers:
            if len(inliers_indices) > len(best_inliers_indices):
                best_inliers_indices = inliers_indices
                best_line = line_params
        
    
    return best_line, best_inliers_indices


def draw_line_on_frame(frame, line_params, color=(0, 255, 0), thickness=2):
    # Unpack line parameters (slope and intercept)
    if line_params is None:
        return  # If line_params is None, there's nothing to draw
    slope, intercept = line_params

    # Ensure slope is not zero to avoid division by zero error
    if np.any(slope== 0):
        slope += 0.0000001

    # Set x1 to zero and calculate corresponding y1 using the intercept
    x1 = 0
    y1 = int(intercept)

    # Keep y2 at the bottom of the frame and calculate corresponding x2
    x2 = frame.shape[1]
    y2 = int(slope * x2 + intercept)


    
    # Draw the line on the frame
    cv2.line(frame, (x1, y1), (x2, y2), color, thickness)


def combine_lines(old_line, n_line, a=0.9):

    if np.any(old_line == None):
        return n_line
    if np.any(n_line == None): 
        return old_line
    
    
    res = np.array([a*old_line[0] + (1-a)*n_line[0],a*old_line[1] + (1-a)*n_line[1]])
    return res


def process_frame(frame):
    Rotated_Cropped_frame = cropLandScapeAndRotate(frame)
    canny_frame = find_edges(Rotated_Cropped_frame)
    line_left_x, line_left_y, line_right_x, line_right_y = find_lines_with_hough_transform(canny_frame, Rotated_Cropped_frame)
    return Rotated_Cropped_frame, line_left_x, line_left_y, line_right_x, line_right_y


def update_line(curr_line, new_line_data):
    new_line = ransac_line_fit(new_line_data[0], new_line_data[1])
    updated_line = combine_lines(curr_line, new_line[0])
    return updated_line


def update_line(curr_line, new_line_data):
    new_line = ransac_line_fit(new_line_data[0], new_line_data[1])
    updated_line = combine_lines(curr_line, new_line[0])
    return updated_line , new_line


def draw_line_if_exists(frame, line, color, thickness):
    if line is not None:
        draw_line_on_frame(frame, line, color=color, thickness=thickness)


def is_swtich_lane(left, right, frame_y_size):

    if np.any(left == None):
        if np.abs(left[0][1]) > frame_y_size // 20:
            return True, True
    if np.any(right == None): 
        if right[0][1] < frame_y_size * 7 // 9:
            return True, False
    return False, False


def process_video_frames(frames):
    """
    Processes each frame from the video, updating and drawing lane lines.
    Parameters:
    - frames: A list of video frames to process.
    Returns:
    - A list of processed frames with lane lines drawn on them.
    """
    n_frames = []  # Initialize the list to store processed frames
    curr_left_line, curr_right_line = None, None  # Initialize current lane lines to None
    is_line_swap = False
    time_counter = 0
    # Loop through each frame in the video
    for frame in tqdm(frames):


        # Rotate and crop the frame, then apply edge detection and find line segments using Hough Transform
        processed_frame, line_left_x, line_left_y, line_right_x, line_right_y = process_frame(frame)

        if is_line_swap:
            time_counter = 1

        if time_counter % 50 != 0:
            time_counter += 1
            is_line_swap = True
        else:
            time_counter = 0

        if is_line_swap and time_counter == 1:
            if is_left_side:
                curr_right_line = n_left[0]
                curr_left_line = None
            else:
                curr_left_line = n_right[0]
                curr_right_line = None
        
        # Update the current left lane line based on detected line segments
        curr_left_line, n_left = update_line(curr_left_line, (line_left_x, line_left_y))
        # Update the current right lane line based on detected line segments
        curr_right_line, n_right = update_line(curr_right_line, (line_right_x, line_right_y))

        is_line_swap, is_left_side = is_swtich_lane(n_left, n_right, frame.shape[0])

        # If a lane line exists, draw it on the frame
        if not is_line_swap:
            draw_line_if_exists(processed_frame, curr_right_line, color=(0, 255, 0), thickness=2)
            draw_line_if_exists(processed_frame, n_left[0], color=(255, 255, 0), thickness=2)
            draw_line_if_exists(processed_frame, curr_left_line, color=(0, 0, 255), thickness=2)
        else:
            if is_left_side:
                draw_line_if_exists(processed_frame, curr_right_line, color=(255, 255, 0), thickness=4)
            else:
                draw_line_if_exists(processed_frame, curr_left_line, color=(0, 255, 255), thickness=4)
                # Check if there is a lane switch
        # Rotate the processed frame back to its original orientation and add it to the list of processed frames
        n_frames.append(cv2.rotate(processed_frame, cv2.ROTATE_90_COUNTERCLOCKWISE))

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


def main():
    video_path = "data/roadCam.mp4"
    frames = extract_frames(video_path, frame_skip=10)
    n_frames = process_video_frames(frames)
    display_frames(n_frames, 250)

if __name__ == "__main__":
    main()
