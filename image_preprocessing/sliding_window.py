import numpy as np
import cv2 as cv


def label_image_watershed(img, contours, indices, topics=10):
    colors = []
    original = np.array(img)

    for i in range(topics):
        color = list(np.random.choice(range(256), size=3))
        colors.append(color)

    index = 0
    for cnt in contours:
        color = (int(colors[indices[index]][0]), int(colors[indices[index]][1]), int(colors[indices[index]][2]))
        cv.drawContours(img, [cnt], 0, color, thickness=-1)
        index = index + 1

    cv.imshow("Segmented", img)
    cv.imshow("Original", original)

    cv.waitKey(0)


def oversegmentation_watershed(img):
    imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    cv.imshow("Grayscale", imgray)
    cv.waitKey(0)

    ret, thresh = cv.threshold(imgray, 75, 255, 0)
    im2, contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    print(len(contours))
    print(contours[0])

    images = []
    usable_contours = []

    MIN_CONTOUR_AREA = 100

    for cnt in contours:
        contour_area = cv.contourArea(cnt)

        if (contour_area < MIN_CONTOUR_AREA):
            continue
        x, y, w, h = cv.boundingRect(cnt)
        ROI = img[y:y + h, x:x + w]
        images.append(ROI)
        usable_contours.append(cnt)

    return images, usable_contours


def split_image(img, n=100, show=False):
    images = []
    for r in range(0, img.shape[0], n):
        for c in range(0, img.shape[1], n):
            images.append(img[r:r + n, c:c + n, :])

            if show:
                cv.imshow("Split Image", img[r:r + n, c:c + n, :])
                cv.waitKey(0)

    return images


def label_image(img, indices, topics=10, n=100):
    colors = []
    x = 0
    original = np.array(img)

    for i in range(topics):
        color = list(np.random.choice(range(256), size=3))
        colors.append(color)
    # print(img.shape)
    for r in range(0, img.shape[0], n):
        for c in range(0, img.shape[1], n):

            if img[r:r + n, c:c + n, :] is not None:
                img[r:r + n, c:c + n, :] = colors[indices[x]]
                x = x + 1

    # cv.imshow("Segmented Image", img)
    # cv.imshow("Original Image", original)

    added_image = cv.addWeighted(img, 0.4, original, 0.1, 0)
    cv.imshow('combined', added_image)

    cv.waitKey(0)
