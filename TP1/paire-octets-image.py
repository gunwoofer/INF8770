 # -*-coding:Latin-1 -* 

import numpy as np
import matplotlib.pyplot as py
import time
import regex as re


print ("Codage paire octets image..")

def rgb2gray(rgb):
    return np.dot(rgb[:,:], [0.299, 0.587, 0.114])

def convertNumber(array):
    for i in range (len(array)):
        if len(array[i]) == 1:
            array[i] = '00' + array[i]
        elif len(array[i]) == 2:
            array[i] = '0' + array[i]

fig1 = py.figure(figsize = (10,10))
imagelue = py.imread('degrade.jpg')
image=imagelue.astype('float')
image=rgb2gray(image)

imageout=image.astype('uint8').flatten().astype("str")
convertNumber(imageout)

Message = ''.join(imageout)

# py.imshow(imageout,cmap = py.get_cmap('gray'))
# py.show()

#Message = "001100011110001010101011110001101010111111111111110000000000000000110101010100011101010101"
#Message =""
#for i in range(33020):
 #   if i < 33020 /2:
  #      Message += '0'
   # else:
    #    Message += '1'
LUToctetsdispo = [True] * 0xffff
dictsymb =[Message[0]]
LUToctetsdispo[ord(Message[0])] = False

nbsymboles = 1
start_time = time.time()

for i in range(1,len(Message)):
    if Message[i] not in dictsymb:
        dictsymb += [Message[i]]
        LUToctetsdispo[ord(Message[i])] = False  #Octet utilisé
        nbsymboles += 1
        
longueurOriginale = np.ceil(np.log2(nbsymboles))*len(Message)

dictsymb = []  #Dictionnaire des substitutions
debut = ord(Message[0])  # Origine trouver un code de substitution. Et pour avoir des caractères imprimables...

remplacementpossible = True
while remplacementpossible == True:
    #Recherche des paires
    paires = []
    for i in range(0,len(Message)-1):
        symbole1, symbole2 = Message[i], Message[i+1]
        temppaire = Message[i] + Message[i+1]
        if not list(filter(lambda x: x[0] == temppaire, paires)): #Si la liste retournee par filter est vide.
            longueur = Message.count(temppaire)
            paires += [[temppaire, longueur]]

    #Trouve la paire avec le plus de repetitions.
    paires = sorted(paires, key=lambda x: x[1], reverse = True)

    if paires[0][1] > 1:
        #Remplace la paire
        print(paires)
        print("La paire ",paires[0][0], " est la plus frequente avec ",paires[0][1], "repetitions")
        #Cherche un octet non utilise
        while debut <0xffff and LUToctetsdispo[debut] == False:
            debut += 1
        if debut < 0xffff:     
            #On substitut
            Message = Message.replace(paires[0][0],  chr(debut))
            LUToctetsdispo[debut] = False
            dictsymb += [[paires[0][0], chr(debut)]]
        else:
            print("Il n y a plus d octets disponible!")
            remplacementpossible = False

    else:
        remplacementpossible = False


temps_execution = time.time() - start_time
print("Temps d execution : ", time.time() - start_time, " secondes")

print("Longueur = {0}".format(np.ceil(np.log2(nbsymboles))*len(Message)))
print("Longueur originale = {0}".format(longueurOriginale))

