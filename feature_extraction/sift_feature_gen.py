import numpy as np
import cv2 as cv

def generate(img, show_keypoints=True):
    '''
    TODO: Create method which takes an image and returns encoded features using SIFT
    '''

    sift = cv.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)
    keypoints = cv.drawKeypoints(img, kp, img)
    if show_keypoints:
        cv.imshow("Keypoints", keypoints)
        cv.waitKey(0)

    return des
