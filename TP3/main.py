import sys
import cv2
import numpy as np

FILENAME = "julia.avi" 

def main():
    print("TP3 INF8770..")
    cap = cv2.VideoCapture()
    cap.open(FILENAME)
 
    if not cap.isOpened():
        print("Erreur pendant l'ouverture de la video %s" % FILENAME)
        return
    else:
        print("Traitement de la video %s..." % FILENAME)
 
    while True:
        (rv, im) = cap.read()   
        if not rv:
            break
        # Traitement image par image
        test = 2
 
    cap.release()
 
 
if __name__ == "__main__":
    main()