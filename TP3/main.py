import sys
import cv2
import numpy as np

FILENAME = "julia.avi" 

def main():
    print("TP3 INF8770..")
    cap = cv2.VideoCapture()
    cap.open(FILENAME)
 
    if not cap.isOpened():
        print("Fatal error - could not open video %s." % FILENAME)
        return
    else:
        print("Parsing video %s..." % FILENAME)
 
    
 
    cap.release()
 
 
if __name__ == "__main__":
    main()