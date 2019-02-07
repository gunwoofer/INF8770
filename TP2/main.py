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
            # if (row % 2 == 0 and col % 2 == 0):
            #     imageycbcr[row][col] = [Y,Cb,Cr]
            # else:
            #     imageycbcr[row][col] = [Y,0,0]
    return imageycbcr

           
def ycbcr2rgb(imageycbcr):
    imagergb = np.zeros_like(imageycbcr)
    for row in range(len(imageycbcr)):
        for col in range(len(imageycbcr[row])):
            # oldi = row
            # oldj = col
            # Y = imageycbcr[row][col][0]
            # if (row % 2 == 0 and col % 2 == 0):
            #     Cb = imageycbcr[row][col][1]
            #     Cr = imageycbcr[row][col][2]
            # else:
            #     if (row % 2 == 1):
            #         oldi = row - 1
            #     if (col % 2 == 1):
            #         oldj = col - 1
            #     Cb = imageycbcr[oldi][oldj][1]
            #     Cr = imageycbcr[oldi][oldj][2]
            Y = imageycbcr[row][col][0]
            Cb = imageycbcr[row][col][1]
            Cr = imageycbcr[row][col][2]
            R = Y + 1.403 * (Cr - 128)
            G = Y - 0.714 * (Cr - 128) - 0.344 * (Cb - 128)
            B = Y + 1.773 * (Cb - 128)
            imagergb[row][col] = [R,G,B]
    return imagergb

image = rgb2ycbcr(image)
image = ycbcr2rgb(image)

py.imshow(image)
py.show()