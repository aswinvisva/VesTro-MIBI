def split_image(img, n=100):
    images = []
    for r in range(0, img.shape[0], n):
        for c in range(0, img.shape[1], n):
            images.append(img[r:r + n, c:c + n, :])

    return images