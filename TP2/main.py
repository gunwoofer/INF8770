import matplotlib.pyplot as py
import numpy as np

print ("TP2 INF8770..")

image = py.imread("image.jpeg")
image = np.array(image)


def rgb2ycbcr(imagergb):
    print ("Etape 1 - Conversion RGB - Y'CbCr..")
    imageycbcr = np.zeros_like(imagergb)
    for row in range(len(imagergb)):
        for col in range(len(imagergb[row])):
            R = imagergb[row][col][0]
            G = imagergb[row][col][1]
            B = imagergb[row][col][2]
            Y = 0.299 * R + 0.587 * G + 0.114 * B
            Cb = 128 + 0.564 * (B - Y)
            Cr = 128 + 0.713 * (R - Y)
            imageycbcr[row][col] = [Y,Cb,Cr]
    return imageycbcr
            

image = rgb2ycbcr(image)


py.imshow(image)
py.show()