import cv2 as cv
import numpy as np
import json

cam_matrix: np.ndarray | None = None
distortion_coeffs: np.ndarray | None = None

def get_values() -> None:
    global cam_matrix, distortion_coeffs
    fs: cv.FileStorage = cv.FileStorage("./calibrations/arducam-calib.json", cv.FileStorage_READ)

    cam_matrix = fs.getNode("camera_matrix").mat()
    distortion_coeffs = fs.getNode("distortion_coefficients").mat()

    fs.release()

def undistort(img: np.ndarray) -> np.ndarray:
    new_image: np.ndarray = np.empty_like(img)
    cv.undistort(img, cam_matrix, distortion_coeffs, new_image)
    return new_image

if __name__ == "__main__":
    get_values()
    print(cam_matrix)
    print(distortion_coeffs)

