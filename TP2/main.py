import matplotlib.pyplot as py
import numpy as np

INDEX_Y = 0
INDEX_CB = 1
INDEX_CR = 2

INDEX_R = 0
INDEX_G = 1
INDEX_B = 2

BLOCK_SIZE = 8
print ("TP2 INF8770..")

image = py.imread("image2.jpeg")
image = np.array(image)
test = 5
def rgb2ycbcr(imagergb):
    print ("Etape 1 - Conversion RGB - Y'CbCr..")
    imageycbcr = np.zeros_like(imagergb)
    imageycbcr = imageycbcr.astype('float64')
    for row in range(imagergb.shape[0]):
        for col in range(imagergb.shape[1]):
            R = imagergb[row][col][INDEX_R]
            G = imagergb[row][col][INDEX_G]
            B = imagergb[row][col][INDEX_B]
            Y = 0.299 * R + 0.587 * G + 0.114 * B
            Cb = 128 + 0.564 * (B - Y)
            Cr = 128 + 0.713 * (R - Y)
            if (row % 2 == 0 and col % 2 == 0):
                if imageycbcr.shape[2] == 4:
                    imageycbcr[row][col] = [Y,Cb,Cr, imagergb[row][col][3]]
                else:
                    imageycbcr[row][col] = [Y,Cb,Cr]
            else:
                if imageycbcr.shape[2] == 4:
                    imageycbcr[row][col] = [Y,0,0, imagergb[row][col][3]]
                else:
                    imageycbcr[row][col] = [Y,0,0]
    return imageycbcr

           
def ycbcr2rgb(imageycbcr):
    imagergb = np.zeros_like(imageycbcr)
    for row in range(imageycbcr.shape[0]):
        for col in range(imageycbcr.shape[1]):
            oldi = row
            oldj = col
            Y = imageycbcr[row][col][0]
            if (row % 2 == 0 and col % 2 == 0):
                Cb = imageycbcr[row][col][INDEX_CB]
                Cr = imageycbcr[row][col][INDEX_CR]
            else:
                if (row % 2 == 1):
                    oldi = row - 1
                if (col % 2 == 1):
                    oldj = col - 1
                Cb = imageycbcr[oldi][oldj][INDEX_CB]
                Cr = imageycbcr[oldi][oldj][INDEX_CR]

            Y = imageycbcr[row][col][INDEX_Y]

            R = Y + (1.403 * (Cr - 128))
            G = Y - (0.714 * (Cr - 128)) - (0.344 * (Cb - 128))
            B = Y + (1.773 * (Cb - 128))
            if imageycbcr.shape[2] == 4:
                imagergb[row][col] = [R,G,B, imageycbcr[row][col][3]]
            else:
                imagergb[row][col] = [R,G,B]
    return imagergb.astype('uint8')

def diff(image, image2):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if (image[i][j] != image2[i][j]).any():
                print( "L'index ({},{}) est différent. Image1 : {}, image2 : {}".format(i, j, image[i][j], image2[i][j]))

def division8x8(image):
    bloc8x8 = np.zeros_like(image)
    bloc8x8.resize(int(bloc8x8.shape[0]/8), int(bloc8x8.shape[1] / 8), 3)
    for row in range(0,image.shape[0] - BLOCK_SIZE, BLOCK_SIZE):
        for column in range(0,image.shape[1] - BLOCK_SIZE, BLOCK_SIZE):
            bloc8x8.append(image[row:row+BLOCK_SIZE,column:column+BLOCK_SIZE])
    return bloc8x8

def inverseDivision8x8(image, result):
    for i in range(0,int(result.shape[0] / BLOCK_SIZE - 1)):
        for j in range(0,int(result.shape[1] / BLOCK_SIZE - 1)):
            result[i*BLOCK_SIZE:i*BLOCK_SIZE+BLOCK_SIZE,j*BLOCK_SIZE:j*BLOCK_SIZE+BLOCK_SIZE,:] = image[i* BLOCK_SIZE+j, :, :]
    return result


image1 = rgb2ycbcr(image)
print('Etape 2 : Division en bloc 8x8')
bloc8x8 = division8x8(image1)

print('Reconstruction de l image')
noBloc = np.zeros_like(image1)
noBloc = inverseDivision8x8(bloc8x8, noBloc)
print('inversion y cb cr')
image2 = ycbcr2rgb(noBloc)

# diff(image, image2)
py.imshow(image2)
py.show()