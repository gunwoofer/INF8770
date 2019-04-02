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

def generateMask(im):
    masques = []
    for i in range(0, 4):
        for j in range(0, 4):        
            masque = np.zeros_like(rgb2gray(im).astype("uint8"))
            masque[int(im.shape[0]*i/4):int(im.shape[0]*(i+1) / 4), int(im.shape[1]*j/4):int(im.shape[1]*(j+1) / 4)] = 1
            masques.append(masque)
    return masques

def generateHist(im, masks):
    hists = []
    for mask in masks:
        hist = cv2.calcHist([im], [0], mask, [256], [0,256])
        hists.append(hist)
    return hists

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
    masques = generateMask(im)

    histoDiffTrame = []
    liste_histo = []
    while rv:
        # Traitement image par image
        images.append(im)
        histQuantified = []
        # im = rgb2gray(im).astype("uint8")

        # Histogrammes
        hists = generateHist(im, masques)

        # Concatenation
        histo = np.concatenate(hists)
        
        
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