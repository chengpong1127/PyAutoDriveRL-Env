import cv2
import numpy as np
def region(image):
    height, width = image.shape
    triangle = np.array([
                    [(0, height / 3), 
                     (width / 2, height / 5), 
                     (width, height / 3)]
                    ], dtype=np.int32)
    
    mask = np.zeros_like(image)
    
    mask = cv2.fillPoly(mask, triangle, 255)
    mask = cv2.bitwise_and(image, mask)
    
    return mask
def lane_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=40, maxLineGap=5)
    
    line_image = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.uint8)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), 255, 5)
    
    return {
        "gray": gray,
        "blur": blur,
        "edges": edges,
        "line_image": line_image
    }
    
