import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

SEUIL_CUT = 55000
SEUIL_FADE = (10000, 50000)
FILENAME = "julia.avi" 
cuts = []
fades = []

def rgb2gray(rgb):
    return np.dot(rgb[:,:], [0.299, 0.587, 0.114])


def diffTrame(histo1, histo2):
    diff = math.sqrt((np.square(histo1 - histo2)).sum())
    return diff

def main():
    print("TP3 INF8770..")
    cap = cv2.VideoCapture()
    cap.open(FILENAME)
 
    if not cap.isOpened():
        print("Erreur pendant l'ouverture de la video %s" % FILENAME)
        return
    else:
        print("Traitement de la video %s..." % FILENAME)
   
    (rv, im) = cap.read()  
    indexTrame = 0
    histoDiffTrame = []

    images = []
    while rv:
        # Traitement image par image
        images.append(im)
        newHist = []
        # im = rgb2gray(im).astype("uint8")

        # Histogrammes
        histoR = cv2.calcHist([im], [0], None, [256], [0,256])
        histoG = cv2.calcHist([im], [1], None, [256], [0,256])
        histoB = cv2.calcHist([im], [2], None, [256], [0,256])

        
        # Concatenation
        histo = np.concatenate((histoR, histoG, histoB))
        
        # Quantification
        for i in range(0,256,8):
            newHist.append(histo[i:i+8].sum())

        # Difference avec la trame precedante
        # newnewHist = np.array(newHist)
        if indexTrame >= 1:
            histoDiffTrame.append(diffTrame(np.array(newHist), np.array(histoPrec)))
        histoPrec = newHist

        (rv, im) = cap.read()  
        indexTrame = indexTrame + 1
        print(indexTrame)

# np.where(np.array(histoDiffTrame) > SEUIL_CUT)
    histoDiffTrame = np.array(histoDiffTrame)
    cuts = np.where(np.array(histoDiffTrame) > SEUIL_CUT)



    for cut in cuts[0]:
        img1 = images[cut]
        img2 = images[cut + 1]
        f = plt.figure()
        f.add_subplot(1,2, 1)
        plt.imshow(img1)
        f.add_subplot(1,2, 2)
        plt.imshow(img2)
        plt.show(block=True)
    plt.plot(histoDiffTrame)
    plt.show()


    cap.release()
 
 
# import numpy as np
# array = np.array([1, 2, 4, 5, 6])
# test2 = array[np.where(array>3)]
# print(test2)


if __name__ == "__main__":
    main()