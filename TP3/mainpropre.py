import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import signal

SEUIL_CUT = 10000
SEUIL_FADE = 6000
FILENAME = "julia.avi" 
cuts = []
fades = []

images = []
fades_by_mean = []
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
        #for _ in range(0, 3): #rgb
        hist = cv2.calcHist([im], [0], mask, [256], [0,256])
        hists.append(hist)
    return hists

def generateHistWithoutMask(im):
    hists = []
    hist = cv2.calcHist([im], [0], None, [256], [0,256])
    hists.append(hist)
    return hists

def regenerateCuts(cuts, trame):
    results = []
    for cut in cuts:
        if int(trame[cut]) > SEUIL_CUT:
            results.append(cut)
    results = np.array(results)
    results[:] += 1
    return results

def convolution(image, matrix):
    newImage = signal.convolve2d(rgb2gray(image).astype("uint8"), matrix)
    for i in range(newImage.shape[0]):
        for j in range(newImage.shape[1]):
            if newImage[i][j] < 127:
                newImage[i][j] = 0
            else:
                newImage[i][j] = 255
    return newImage

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

        # Histogrammes
        hists = generateHist(im, masques)
        # Concatenation
        histo = np.concatenate(hists)
        
        # Quantification
        niveau = 32 # 64 on perd des cuts + 32 on a des erreurs dans fades, mais on a tous les cut
        for i in range(0,histo.shape[0],niveau):
            histQuantified.append(histo[i:i+niveau].sum())
        liste_histo.append(histQuantified)
         

        # Difference avec la trame precedante
        
        if indexTrame > 0:
            histoDiffTrame.append(abs(diffTrame(np.array(liste_histo[indexTrame]), np.array(liste_histo[indexTrame - 1]))))
        #if indexTrame > 1:
         #   histoDiffTrame.append(abs(diffTrame2(np.array(liste_histo[indexTrame]), np.array(liste_histo[indexTrame - 1]), liste_histo[indexTrame - 2])))
        
        (rv, im) = cap.read()  
        indexTrame = indexTrame + 1

    histoDiffTrame = np.array(histoDiffTrame)
    cuts = np.where(np.array(histoDiffTrame) > SEUIL_FADE)
    real_cuts = []
    real_fades = []

    for i in range(0, cuts[0].shape[0] - 1):
        cut = int(cuts[0][i])
        cut_next = int(cuts[0][i+1])
        if i > 0:
            cut_prev = int(cuts[0][i-1])
            if cut - cut_prev == 1 :
                real_fades.append(cut)
            elif cut_next - cut == 1:
                real_fades.append(cut)
            else:
                real_cuts.append(cut)
        else:
            if cut_next - cut == 1:
                real_fades.append(cut)
            else:
                real_cuts.append(cut)
    last_cut = int(cuts[0][i+1])
    cut_prev = int(cuts[0][i])
    if last_cut - cut_prev == 1 :
        real_fades.append(last_cut)
    else:
        real_cuts.append(last_cut)

    real_cuts = regenerateCuts(real_cuts, histoDiffTrame)
    # Affichage des cuts
    for cut in real_cuts:
        # img1 = images[cut]
        # img2 = images[cut + 1]
        # f = plt.figure()
        # f.add_subplot(1,2, 1)
        # plt.imshow(img1)
        # f.add_subplot(1,2, 2)
        # plt.imshow(img2)
        print("cut : " + str(cut))
        # plt.show(block=True)

    for fade in real_fades:
    #     img1 = images[fade]
    #     img2 = images[fade + 1]
    #     f = plt.figure()
    #     f.add_subplot(1,2, 1)
    #     plt.imshow(img1)
    #     f.add_subplot(1,2, 2)
    #     plt.imshow(img2)
        print("fade : " + str(fade))
    #     plt.show(block=True)

    plt.plot(histoDiffTrame)
    plt.show()


    cap.release()
 



if __name__ == "__main__":
    main()