import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

SEUIL_CUT = 8000
SEUIL_FADE = (10000, 50000)
FILENAME = "julia.avi" 
cuts = []
fades = []

images = []

def rgb2gray(rgb):
    return np.dot(rgb[:,:], [0.299, 0.587, 0.114])


def diffTrame(histo1, histo2):
    diff = math.sqrt((np.square(histo1 - histo2)).sum())
    return diff

def diffTrame2(histo1, histo2, histo3):
    return diffTrame(histo3, histo2) - diffTrame(histo2, histo1)

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

    # Masques
    masqueHG = np.zeros_like(rgb2gray(im).astype("uint8"))
    masqueHG[:int(im.shape[0]/2), :int(im.shape[1]/2)] = 1
    masqueBG = np.zeros_like(rgb2gray(im).astype("uint8"))
    masqueBG[:int(im.shape[0]/2), int(im.shape[1]/2):] = 1
    masqueHD = np.zeros_like(rgb2gray(im).astype("uint8"))
    masqueHD[int(im.shape[0]/2):, :int(im.shape[1]/2)] = 1
    masqueBD = np.zeros_like(rgb2gray(im).astype("uint8"))
    masqueBD[int(im.shape[0]/2):, int(im.shape[1]/2):] = 1

    histoDiffTrame = []
    liste_histo = []
    while rv:
        # Traitement image par image
        images.append(im)
        histQuantified = []
        im = rgb2gray(im).astype("uint8")

        # Histogrammes
        histoHG = cv2.calcHist([im], [0], masqueHG, [256], [0,256])
        histoBG = cv2.calcHist([im], [0], masqueBG, [256], [0,256])
        histoHD = cv2.calcHist([im], [0], masqueHD, [256], [0,256])
        histoBD = cv2.calcHist([im], [0], masqueBD, [256], [0,256])

        # Concatenation
        histo = np.concatenate((histoHG, histoBG, histoHD, histoBD))
        
        
        # Quantification
        for i in range(0,histo.shape[0],8):
            histQuantified.append(histo[i:i+8].sum())
        liste_histo.append(histQuantified)
        # Difference avec la trame precedante
        if indexTrame > 1:
            # histoDiffTrame.append(diffTrame(np.array(histQuantified), np.array(histoPrec)))
            histoDiffTrame.append(diffTrame2(np.array(liste_histo[indexTrame]), np.array(liste_histo[indexTrame - 1]), liste_histo[indexTrame - 2]))

        histoPrec = histQuantified
        

        (rv, im) = cap.read()  
        indexTrame = indexTrame + 1
        #print(indexTrame)

    histoDiffTrame = np.array(histoDiffTrame)
    cuts = np.where(np.array(histoDiffTrame) > SEUIL_CUT)


    img1 = images[345]
    img2 = images[350 + 1]
    f = plt.figure()
    f.add_subplot(1,2, 1)
    plt.imshow(cv2.cvtColor(img1,cv2.COLOR_BGR2RGB))
    f.add_subplot(1,2, 2)
    plt.imshow(cv2.cvtColor(img2,cv2.COLOR_BGR2RGB))
    plt.show(block=True)
    # Affichage des cuts
    for cut in cuts[0]:
        img1 = images[cut]
        img2 = images[cut + 1]
        f = plt.figure()
        f.add_subplot(1,2, 1)
        plt.imshow(img1)
        f.add_subplot(1,2, 2)
        plt.imshow(img2)
        print("Image numero : " + str(cut))
        plt.show(block=True)
    plt.plot(histoDiffTrame)
    plt.show()


    cap.release()
 



if __name__ == "__main__":
    main()