import numpy as np
import cv2 as cv

'''
Author: Aswin Visva
Email: aavisva@uwaterloo.ca
'''


class SIFTFeatureGen:

    def __init__(self, show_keypoints=False, n_keypoints=32):
        self.show_keypoints = show_keypoints
        self.n_keypoints = n_keypoints

    def generate(self, images):
        y = []

        for img in images:
            sift = cv.xfeatures2d.SIFT_create(self.n_keypoints)
            kp, des = sift.detectAndCompute(img, None)
            if self.show_keypoints:
                keypoints = cv.drawKeypoints(img, kp, img)
                cv.imshow("Keypoints", keypoints)
                cv.waitKey(0)

            if des is None:
                keypoints = cv.drawKeypoints(img, kp, img)
                cv.imshow("Keypoints", keypoints)
                cv.waitKey(0)

            y.append(des)

        return y
