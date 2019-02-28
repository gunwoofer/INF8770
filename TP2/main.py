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

# https://stackoverflow.com/questions/34913005/color-space-mapping-ycbcr-to-rgb
def rgb2ycbcr(im):
    xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    ycbcr = im.dot(xform.T)
    ycbcr[:,:,[1,2]] += 128
    return np.uint8(ycbcr)

def chrominance(im):
    for row in range(im.shape[0]):
        for col in range(im.shape[1]):
            oldi = row
            oldj = col
            if (row % 2 == 1):
                oldi = row - 1
            if (col % 2 == 1):
                oldj = col - 1
            im[row][col][INDEX_CB] = im[oldi][oldj][INDEX_CB]
            im[row][col][INDEX_CR] = im[oldi][oldj][INDEX_CR]

# https://stackoverflow.com/questions/34913005/color-space-mapping-ycbcr-to-rgb
def ycbcr2rgb(im):
    xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    rgb = im.astype(np.float)
    rgb[:,:,[1,2]] -= 128
    rgb = rgb.dot(xform.T)
    np.putmask(rgb, rgb > 255, 255)
    np.putmask(rgb, rgb < 0, 0)
    return np.uint8(rgb)

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
print('Etape 1.5 : Appliquer la chrominance')
chrominance(image1)
print('Etape 2 : Division en bloc 8x8')
# bloc8x8 = division8x8(image1)

# print('Reconstruction de l image')
# noBloc = np.zeros_like(image1)
# noBloc = inverseDivision8x8(bloc8x8, noBloc)
print('inversion y cb cr')
image2 = ycbcr2rgb(image1)

# diff(image, image2)
py.imshow(image2)
py.show()