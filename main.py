import os
import cv2
import time
import pytesseract
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def display_image(im):
    """
    Display an image using Matplotlib.

    Args:
    - im: Image data as a NumPy array or file path.

    """
    dpi = 150
    
    if isinstance(im, str) and os.path.isfile(im):
        im_data = plt.imread(im)
    elif isinstance(im, (np.ndarray, np.generic)):
        im_data = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError("Input must be a file path or image data array.")


    height, width  = im_data.shape[:2]
    
    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')

    # Display the image.
    ax.imshow(im_data, cmap='gray')
    plt.show()


def preprocess_image(image):
    """
    Preprocess the input image.

    Args:
    - image: Input image data as a NumPy array.

    Returns:
    - Preprocessed image data (grayscale and thresholded).

    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    return (gray, thresh)


def find_grid(image_thresh):
    """
    Find the grid lines in the preprocessed image.

    Args:
    - image_thresh: Thresholded image data.

    Returns:
    - Image data with grid lines detected.

    """
    kernel_len = np.array(image_thresh).shape[1] // 100
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    vertical_lines = cv2.erode(image_thresh, ver_kernel, iterations=3)
    vertical_lines = cv2.dilate(vertical_lines, ver_kernel, iterations=3)

    horizontal_lines = cv2.erode(image_thresh, hor_kernel, iterations=3)
    horizontal_lines = cv2.dilate(horizontal_lines, hor_kernel, iterations=3)

    grid_vh = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0)
    grid_vh = cv2.erode(~grid_vh, kernel, iterations=2)
    grid_vh = cv2.threshold(grid_vh, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    return grid_vh


def sort_contours(cnts, method="left-to-right"):
        """
        Sort contours based on the provided method.

        Args:
        - cnts: List of contours to sort.
        - method: Sorting method (left-to-right, right-to-left, top-to-bottom, bottom-to-top).

        Returns:
        - Sorted list of contours.
            
        """
        reverse = False
        i = 0
        if method == "right-to-left" or method == "bottom-to-top":
            reverse = True
        if method == "top-to-bottom" or method == "bottom-to-top":
            i = 1
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))
        return (cnts, boundingBoxes)


def xor_and_not_grid(gray, grid):
    """
    Apply XOR and NOT operations on the grid with the original image.

    Args:
    - gray: Input grayscale image data as a NumPy array.
    - grid: Image data with grid lines.

    Returns:
    - Processed image data.

    """
    grid_bitxor = cv2.bitwise_xor(gray, grid)
    grid_bitnot = cv2.bitwise_not(grid_bitxor)
    return grid_bitnot


def extract_cells_from_grid(grid_vh):
    """
    Extract cells from the grid.

    Args:
    - grid_vh: Image data with grid lines.

    Returns:
    - List of boxes containing cell coordinates.

    """

    contours, _ = cv2.findContours(grid_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours, boundingBoxes = sort_contours(contours, method="top-to-bottom")

    heights = [boundingBoxes[i][3] for i in range(len(boundingBoxes))]
    mean_height = np.mean(heights)

    box = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w < 1000 and h < 500:
            box.append([x, y, w, h])

    row = []
    column = []
    j = 0
    for i in range(len(box)):
        if i == 0:
            column.append(box[i])
            previous = box[i]
        else:
            if box[i][1] <= previous[1] + mean_height / 2:
                column.append(box[i])
                previous = box[i]
                if i == len(box) - 1:
                    row.append(column)
            else:
                row.append(column)
                column = []
                previous = box[i]
                column.append(box[i])

    # Calculate maximum number of cells
    countcol = 0
    for i in range(len(row)):
        countcol = len(row[i])
        if countcol > countcol:
            countcol = countcol

    # Retrieve the center of each column
    center = [int(row[i][j][0] + row[i][j][2] / 2) for j in range(len(row[i])) if row[0]]
    center = np.array(center)
    center.sort()

    # Arrange the boxes in respective order
    cells = []
    for i in range(len(row)):
        lis = []
        for k in range(countcol):
            lis.append([])
        for j in range(len(row[i])):
            diff = abs(center - (row[i][j][0] + row[i][j][2] / 4))
            minimum = min(diff)
            indexing = list(diff).index(minimum)
            lis[indexing].append(row[i][j])
        cells.append(lis)

    return cells


def extract_text_from_cells(cells, grid, progress_callback=None):
    """
    Extract text from each cell in the grid.

    Args:
    - cells: List of boxes containing cell coordinates.
    - grid: Image data (must be grayscale) with grid lines.
    - progress_callback: Callback function to track progress.

    Returns:
    - Extracted text as a pandas DataFrame.

    """

    outer = []
    total_cells = len(cells) * len(cells[0])
    current_cell = 0

    for i in range(len(cells)):
        for j in range(len(cells[i])):
            inner = ""
            if len(cells[i][j]) == 0:
                outer.append(' ')
            else:
                for k in range(len(cells[i][j])):
                    y, x, w, h = cells[i][j][k][0], cells[i][j][k][1], cells[i][j][k][2], cells[i][j][k][3]
                    final_img = grid[x:x+h, y:y+w]
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
                    border = cv2.copyMakeBorder(final_img, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[255, 255])
                    resizing = cv2.resize(border, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                    dilation = cv2.dilate(resizing, kernel, iterations=1)
                    erosion = cv2.erode(dilation, kernel, iterations=1)

                    out = pytesseract.image_to_string(erosion)
                    if len(out) == 0:
                        out = pytesseract.image_to_string(erosion, config='--psm 3')
                    inner = inner + " " + out

                    current_cell += 1
                    if progress_callback is not None:
                        progress = int(45 + (current_cell / total_cells) * 55)
                        progress_callback(progress)

                outer.append(inner.strip())

    arr = np.array(outer)
    dataframe = pd.DataFrame(arr.reshape(len(cells), len(cells[0])))
    return dataframe


def parse_timetable(image_path = None, image = None, progress_callback = None):
    """
    Process an image and extract text from the grid.

    Args:
    - image_path: File path of the input image.
    - image: Image data as a NumPy array.
    - progress_callback: Callback function to track progress.

    Returns:
    - Extracted text as a pandas DataFrame.

    """
    if image_path is None and image is None:
        raise ValueError("Please provide an image path or image data.")
    elif image is not None and image_path is not None:
        raise ValueError("Please provide only one of image path or image data.")
    elif image is not None and image_path is None:
        image = cv2.imdecode(np.fromstring(image.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    elif image is None:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Failed to load image from path:", image_path)

    # image = cv2.imread(image_path)
    # if image is None:
    #     raise ValueError("Failed to load image from path:", image_path)

    if progress_callback is not None:
        progress_callback(0)
        time.sleep(0.05)
        gray, image_thresh = preprocess_image(image)
        progress_callback(15)
        grid = find_grid(image_thresh)
        progress_callback(30)
        cells = extract_cells_from_grid(grid)
        progress_callback(45)
        grid = xor_and_not_grid(gray, grid)
        extracted_text = extract_text_from_cells(cells, grid, progress_callback)
        progress_callback(100)
    else:
        gray, image_thresh = preprocess_image(image)
        grid = find_grid(image_thresh)
        cells = extract_cells_from_grid(grid)
        grid = xor_and_not_grid(gray, grid)
        extracted_text = extract_text_from_cells(cells, grid)
    
    return extracted_text

if __name__ == "__main__":
    # Example usage
    image_path = "./dataset/timetable1.png"
    result_df = parse_timetable(image_path)
    print(result_df)