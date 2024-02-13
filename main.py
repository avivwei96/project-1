import cv2
import os
import numpy as np
import warnings
from tqdm import tqdm


def extract_frames(video_path):
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
            # Convert the frame to grayscale or manipulate as needed
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Append the frame to the list
            if i % 5 == 0:
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
    cropped_frame = frame[(height * 3) // 5:, width // 5:(width * 4) // 5]
    
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
            slope = (y2 - y1) / (x2 - x1 + 1e-6).
            
            if y1 > frame.shape[0]//2:
                if -1.8 < slope < 0:
                    line_right_x.extend([x1, x2])
                    line_right_y.extend([y1, y2])
            else:
                if 0 < slope < 1.8:
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
    
    slope += 0.0000001
    # Set x1 to zero and calculate corresponding y1 using the intercept
    x1 = 0
    y1 = int(intercept)

    # Keep y2 at the bottom of the frame and calculate corresponding x2
    x2 = frame.shape[1]
    y2 = int(slope * x2 + intercept)


    
    # Draw the line on the frame
    cv2.line(frame, (x1, y1), (x2, y2), color, thickness)


def combine_lines(old_line, n_line, a=0.7):

    if np.any(old_line == None):
        return n_line
    if np.any(n_line == None): 
        return old_line
    
    
    res = np.array([a*old_line[0] + (1-a)*n_line[0],a*old_line[1] + (1-a)*n_line[1]])
    return res


def line_switch(frame):
    pass

def main():
    # Path to the video file
    video_path = "data/roadCam.mp4"

    # Extract frames from the video
    frames = extract_frames(video_path)

    n_frames = []
    curr_left_line = None
    curr_right_line = None

    # Process the frames (e.g., display, save, etc.)
    for frame in tqdm(frames):
        Rotated_Cropped_frame = cropLandScapeAndRotate(frame)
        canny_frame = find_edges(Rotated_Cropped_frame)
        line_left_x,  line_left_y, line_right_x, line_right_y = find_lines_with_hough_transform(canny_frame, Rotated_Cropped_frame)

        line_left = ransac_line_fit(line_left_x, line_left_y)
        curr_left_line = combine_lines(curr_left_line, line_left[0])
        if curr_left_line is not None:
            draw_line_on_frame(Rotated_Cropped_frame, curr_left_line, color=(0, 255, 255), thickness=2)

        line_right = ransac_line_fit(line_right_x, line_right_y)

        curr_right_line = combine_lines(curr_right_line, line_right[0])
        if curr_left_line is not None: 
            draw_line_on_frame(Rotated_Cropped_frame, curr_right_line, color=(0, 255, 0), thickness=2)

        n_frames.append(cv2.rotate(Rotated_Cropped_frame, cv2.ROTATE_90_COUNTERCLOCKWISE))


    for frame in n_frames:
        cv2.imshow('Frame', frame)  # Provide a window name for display
        if cv2.waitKey(125) & 0xFF == ord('q'):
            break  

    # Close all OpenCV windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
