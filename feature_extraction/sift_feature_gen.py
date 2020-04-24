import numpy as np
import cv2 as cv

def generate(img, show_keypoints=True, n_keypoints=300):
    '''
    TODO: Create method which takes an image and returns encoded features using SIFT
    '''

    sift = cv.xfeatures2d.SIFT_create(n_keypoints)
    kp, des = sift.detectAndCompute(img, None)
    if show_keypoints:
        keypoints = cv.drawKeypoints(img, kp, img)
        cv.imshow("Keypoints", keypoints)
        cv.waitKey(0)

    return des
