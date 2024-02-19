import cv2
import os
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


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

def cropLandScapeAndRotate(frame, height_top=0.6, height_bot=0.9, width_l=0.15, width_r=0.8):
    # Crop the frame to keep only the relevent part
    height, width = frame.shape[:2]
    cropped_frame = frame[int(height * height_top):int(height * height_bot) , int(width * width_l):int(width * width_r)].copy()
    frame[int(height * height_top):int(height * height_bot) , int(width * width_l):int(width * width_r)] = [0, 0, 0]
    
    # Rotate the cropped frame by 180 degrees
    rotated_frame = cv2.rotate(cropped_frame, cv2.ROTATE_90_CLOCKWISE)
    
    return rotated_frame


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


def paste_the_frame(frame, procesed_frame,top_crop_ratio, bottom_crop_ratio, right_crop_ratio):
    rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    # Calculate the height to crop from the top and bottom
    top_crop = int(rotated_frame.shape[0] * top_crop_ratio / 18)  # Adjusted cropping height for the top
    bottom_crop = int(rotated_frame.shape[0] * bottom_crop_ratio / 18)  # Adjusted cropping height for the bottom

    # Calculate the width to crop from the right
    right_crop = int(rotated_frame.shape[1] * right_crop_ratio / 10)  # Adjusted cropping width for the right

    # Crop the frame to remove portions from the top, bottom, and right
    rotated_frame[top_crop:-bottom_crop, :-right_crop] = procesed_frame

    return cv2.rotate(rotated_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)


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
    dilated_frame = cv2.dilate(blurred_frame, kernel, iterations=3)
    erode_frame = cv2.erode(dilated_frame, kernel, iterations=2)


    dilated_frame = crop_right_corners(erode_frame, corner_size=30)

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


# def draw_quadratic_polynomial_on_frame(frame, edges):
#     # Get coordinates of edge points
#     points = np.argwhere(edges > 0)
#     if len(points) == 0:
#         return  # No edge points found, no need to draw anything

#     y = points[:, 0]
#     x = points[:, 1]

#     # Fit a polynomial of degree 2 (quadratic polynomial)
#     poly_fit = np.polyfit(x, y, 3)

#     # Generate x values for plotting the polynomial curve
#     x_values = np.linspace(0, frame.shape[1], 100)
#     y_values = np.polyval(poly_fit, x_values)  # Calculate corresponding y values

#     # Create a list of points for the polynomial curve
#     curve_points = np.column_stack((x_values, y_values)).astype(np.int32)

#     # Create a blank image to draw the polynomial curve
#     curve_image = np.zeros_like(edges)  # Use edges shape to create a grayscale image

#     # Draw the polynomial curve on the blank image
#     cv2.polylines(curve_image, [curve_points], isClosed=False, color=(255, 255, 255), thickness=2)

#     # Convert the grayscale curve image to BGR format
#     curve_image_bgr = cv2.cvtColor(curve_image, cv2.COLOR_GRAY2BGR)

#     # Combine the original frame with the polynomial curve image only if there are dots to display
#     result = frame.copy()
#     if np.sum(curve_image) > 0:  # Check if there are any non-zero pixels in the curve_image
#         result = cv2.addWeighted(frame, 1, curve_image_bgr, 0.5, 0)

#     #cv2.imshow('Frame', result)
#     #cv2.waitKey(100)
#     return result


def draw_quadratic_polynomial_on_frame(frame, edges):
    # Get coordinates of edge points
    points = np.argwhere(edges > 0)
    if len(points) == 0:
        return  # No edge points found, no need to draw anything

    y = points[:, 0]
    x = points[:, 1]

    # RANSAC polynomial fitting
    model = make_pipeline(PolynomialFeatures(degree=3), RANSACRegressor(residual_threshold=35))
    model.fit(x.reshape(-1, 1), y)
    
    # Generate x values for plotting the polynomial curve
    x_values = np.linspace(0, frame.shape[1], 100)
    y_values = model.predict(x_values.reshape(-1, 1))

    # Create a list of points for the polynomial curve
    curve_points = np.column_stack((x_values, y_values)).astype(np.int32)

    # Create a blank image to draw the polynomial curve
    curve_image = np.zeros_like(edges)  # Use edges shape to create a grayscale image

    # Draw the polynomial curve on the blank image
    cv2.polylines(curve_image, [curve_points], isClosed=False, color=(255, 255, 255), thickness=6)

    # Convert the grayscale curve image to BGR format
    curve_image_bgr = cv2.cvtColor(curve_image, cv2.COLOR_GRAY2BGR)

    # Combine the original frame with the polynomial curve image only if there are dots to display
    result = frame.copy()
    if np.sum(curve_image) > 0:  # Check if there are any non-zero pixels in the curve_image
        result = cv2.addWeighted(frame, 1, curve_image_bgr, 0.5, 0)

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

    left_res = draw_quadratic_polynomial_on_frame(crop_rotated_frame_left, canny_frame_left)
    right_res = draw_quadratic_polynomial_on_frame(crop_rotated_frame_right, canny_frame_right)

    res = paste_the_frame(frame, left_res, 5, 8, 6)
    res = paste_the_frame(res, right_res, 10, 5, 6)

    return res

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
    # Loop through each frame in the video
    for frame in tqdm(frames):
        # Rotate and crop the frame, then apply edge detection and find line segments using Hough Transform
        n_frames.append(process_frame(frame))

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
    video_path = "curve/data/curveRoadCam.mp4"
    frames = extract_frames(video_path, frame_skip=1)
    n_frames = process_video_frames(frames)
    save_frames_as_video(n_frames, "data/curve_project_res.mp4", fps=25)


if __name__ == "__main__":
    main()
