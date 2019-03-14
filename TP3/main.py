import sys
import cv2
import numpy as np

FILENAME = "julia.avi" 
cuts = []
fades = []

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
    while rv:
        # Traitement image par image
        histoR = im[:,:,0]
        (rv, im) = cap.read()  

        # Histogrammes
        histoR = cv2.calcHist([im], [0], None, [256], [0,256])
        histoG = cv2.calcHist([im], [1], None, [256], [0,256])
        histoB = cv2.calcHist([im], [2], None, [256], [0,256])
        
        # Concatenation
        histo = np.concatenate((histoR, histoG, histoB))



    cap.release()
 
 
if __name__ == "__main__":
    main()