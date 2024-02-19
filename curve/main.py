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


def cropLandScapeAndRotate(frame, top_crop_ratio, bottom_crop_ratio, right_crop_ratio):
    rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    # Calculate the height to crop from the top and bottom
    top_crop = int(rotated_frame.shape[0] * top_crop_ratio / 18)  # Adjusted cropping height for the top
    bottom_crop = int(rotated_frame.shape[0] * bottom_crop_ratio / 18)  # Adjusted cropping height for the bottom

    # Calculate the width to crop from the right
    right_crop = int(rotated_frame.shape[1] * right_crop_ratio / 10)  # Adjusted cropping width for the right

    # Crop the frame to remove portions from the top, bottom, and right
    cropped_frame = rotated_frame[top_crop:-bottom_crop, :-right_crop]

    return cropped_frame


def paste_the_frame(frame, processed_part):
    height, width = frame.shape[:2]
    frame[int(height * 3.5) // 5:(height * 9) // 10, (width * 3) // 10:(width - (width * 3) // 10)] = processed_part


def find_yellow(frame):
    # Convert BGR image to HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define range of yellow color in HSV
    lower_yellow = np.array([20, 100, 100])  # Lower bound for yellow
    upper_yellow = np.array([30, 255, 255])  # Upper bound for yellow

    # Threshold the HSV image to get only yellow colors
    yellow_mask = cv2.inRange(hsv_frame, lower_yellow, upper_yellow)

    # Bitwise-AND mask and original image
    yellow_result = cv2.bitwise_and(frame, frame, mask=yellow_mask)

    return yellow_result


def find_edges(frame, threshold_value_min=120, threshold_value_max=200):
    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to convert the image to black and white
    _, black_and_white_frame = cv2.threshold(gray_frame, threshold_value_min, threshold_value_max, cv2.THRESH_BINARY)

    # Apply Gaussian blur to reduce noise in the binary image
    blurred_frame = cv2.GaussianBlur(black_and_white_frame, (5, 5), 0)

    # Define a kernel for the dilation. You can adjust the size as needed.
    kernel = np.ones((3, 3), np.uint8)

    # Apply dilation to the blurred binary image
    erode_frame = cv2.erode(blurred_frame, kernel, iterations=1)
    dilated_frame = cv2.dilate(erode_frame, kernel, iterations=1)

    dilated_frame = crop_right_corners(dilated_frame, corner_size=50)

    # Apply Canny edge detection on the dilated black and white image
    edges = cv2.Canny(dilated_frame, 50, 150)

    #cv2.imshow('Frame', black_and_white_frame)
    #cv2.waitKey(100)
    return edges


def crop_right_corners(image, corner_size=50):
    # Create a mask filled with ones (white) with the same dimensions as the image
    mask = np.ones_like(image) * 255

    # Get the dimensions of the image
    height, width = image.shape[:2]

    # Draw a black line to create a diagonal on the top right corner
    cv2.line(mask, (width - corner_size, 0), (width, corner_size), (0, 0, 0), thickness=10)

    # Draw a black line to create a diagonal on the bottom right corner
    cv2.line(mask, (width - corner_size, height), (width, height - corner_size), (0, 0, 0), thickness=10)

    # Fill below the diagonal line for the top right corner to make it solid
    for i in range(corner_size):
        cv2.line(mask, (width - i, 0), (width, i), (0, 0, 0), thickness=1)

    # Fill above the diagonal line for the bottom right corner to make it solid
    for i in range(corner_size):
        cv2.line(mask, (width - i, height), (width, height - i), (0, 0, 0), thickness=1)

    # Apply the mask to the image using bitwise_and to keep the central and left parts unchanged
    cropped_image = cv2.bitwise_and(image, mask.astype(np.uint8))

    return cropped_image


def find_lines_with_hough_transform(edges, frame, hough_threshold=30):
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=hough_threshold, minLineLength=25, maxLineGap=10)
    line_left_x, line_left_y, line_right_x, line_right_y = [], [], [], []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1 + 1e-6)

            if y1 > frame.shape[0] // 2:
                if -1.8 < slope < 0:
                    line_right_x.extend([x1, x2])
                    line_right_y.extend([y1, y2])
            else:
                if 0 < slope < 2.1:
                    line_left_x.extend([x1, x2])
                    line_left_y.extend([y1, y2])

    return np.array(line_left_x), np.array(line_left_y), np.array(line_right_x), np.array(line_right_y)


def ransac_line_fit(x, y, threshold=15, num_iterations=1000, min_inliers=2):
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
            line_params = np.polyfit(sample_x, sample_y, 3)
            line_function = np.poly1d(line_params)

            # Calculate distances of all points to the line
            distances = np.abs(line_function(x) - y)

            # Find inliers (points closer than threshold)
            inliers_indices = np.where(distances < threshold)[0]
            if len(inliers_indices) >= min_inliers:
                if len(inliers_indices) > len(best_inliers_indices):
                    best_inliers_indices = inliers_indices
                    best_line = line_params
        except np.linalg.LinAlgError:
            pass  # Ignore errors from np.polyfit and continue

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


def draw_quadratic_polynomial_on_frame(frame, edges):
    # Get coordinates of edge points
    points = np.argwhere(edges > 0)
    if len(points) == 0:
        return  # No edge points found, no need to draw anything

    y = points[:, 0]
    x = points[:, 1]

    # Fit a polynomial of degree 2 (quadratic polynomial)
    poly_fit = np.polyfit(x, y, 3)

    # Generate x values for plotting the polynomial curve
    x_values = np.linspace(0, frame.shape[1], 100)
    y_values = np.polyval(poly_fit, x_values)  # Calculate corresponding y values

    # Create a list of points for the polynomial curve
    curve_points = np.column_stack((x_values, y_values)).astype(np.int32)

    # Create a blank image to draw the polynomial curve
    curve_image = np.zeros_like(edges)  # Use edges shape to create a grayscale image

    # Draw the polynomial curve on the blank image
    cv2.polylines(curve_image, [curve_points], isClosed=False, color=(255, 255, 255), thickness=2)

    # Convert the grayscale curve image to BGR format
    curve_image_bgr = cv2.cvtColor(curve_image, cv2.COLOR_GRAY2BGR)

    # Combine the original frame with the polynomial curve image only if there are dots to display
    result = frame.copy()
    if np.sum(curve_image) > 0:  # Check if there are any non-zero pixels in the curve_image
        result = cv2.addWeighted(frame, 1, curve_image_bgr, 0.5, 0)

    #cv2.imshow('Frame', result)
    #cv2.waitKey(100)
    return result



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
    crop_rotated_frame_right = cropLandScapeAndRotate(frame, 10, 5, 6)
    crop_rotated_frame_left = cropLandScapeAndRotate(frame, 5, 8, 6)

    frame_left = find_yellow(crop_rotated_frame_left)
    canny_frame_right = find_edges(crop_rotated_frame_right)
    canny_frame_left = find_edges(frame_left,3,255)

    line_left_x, line_left_y, line_right_x, line_right_y = find_lines_with_hough_transform(canny_frame_right, crop_rotated_frame_right)

    #cv2.imshow('Frame', canny_frame_left)
    #cv2.waitKey(100)
    left_res = draw_quadratic_polynomial_on_frame(crop_rotated_frame_left, canny_frame_left)
    right_res = draw_quadratic_polynomial_on_frame(crop_rotated_frame_right, canny_frame_right)

    return left_res, right_res


def update_line(curr_line, new_line_data):
    new_line = ransac_line_fit(new_line_data[0], new_line_data[1])
    updated_line = combine_lines(curr_line, new_line[0])
    return updated_line


def update_line(curr_line, new_line_data):
    new_line = ransac_line_fit(new_line_data[0], new_line_data[1])
    updated_line = combine_lines(curr_line, new_line[0])
    return updated_line, new_line


def draw_line_if_exists(frame, line, color, thickness):
    if line is not None:
        draw_line_on_frame(frame, line, color=color, thickness=thickness)


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
    # Loop through each frame in the video
    for frame in tqdm(frames):
        # Rotate and crop the frame, then apply edge detection and find line segments using Hough Transform
        res_left, res_right = process_frame(frame)





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
    video_path = "data/curveRoadCam.mp4"
    frames = extract_frames(video_path, frame_skip=5)
    n_frames = process_video_frames(frames)
    display_frames(n_frames, 100)


if __name__ == "__main__":
    main()