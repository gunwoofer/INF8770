import numpy as np
import matplotlib.pyplot as py
import time

print ("Codage paire octets image..")

def rgb2gray(rgb):
    return np.dot(rgb[:,:], [0.299, 0.587, 0.114])

fig1 = py.figure(figsize = (10,10))
imagelue = py.imread('RGB.jpg')
image=imagelue.astype('float')
image=rgb2gray(image)
imageout=image.astype('uint8').flatten().astype("character")

# py.imshow(imageout,cmap = py.get_cmap('gray'))
# py.show()


LUToctetsdispo = [True] * 256
dictsymb =[imageout[0]]
LUToctetsdispo[imageout[0].astype('uint8')] = False
nbsymboles = 1
for i in range(1,len(imageout)):
    if imageout[i] not in dictsymb:
        dictsymb += [imageout[i]]
        LUToctetsdispo[imageout[i].astype('uint8')] = False  #Octet utilisé
        nbsymboles += 1
        
longueurOriginale = np.ceil(np.log2(nbsymboles))*len(imageout)

dictsymb = []  #Dictionnaire des substitutions
debut = imageout[0]  # Origine trouver un code de substitution. Et pour avoir des caractères imprimables...

remplacementpossible = True
while remplacementpossible == True:
    #Recherche des paires
    paires = []
    for i in range(0,len(imageout)-1):
        temppaire = imageout[i]+imageout[i+1]
        if not list(filter(lambda x: x[0] == temppaire, paires)): #Si la liste retournée par filter est vide.
            paires += [[temppaire,len(re.findall(temppaire, imageout, overlapped = True))]]

    #Trouve la paire avec le plus de répétitions.
    paires = sorted(paires, key=lambda x: x[1], reverse = True)

    if paires[0][1] > 1:
        #Remplace la paire
        print(paires)
        print("La paire ",paires[0][0], " est la plus fréquente avec ",paires[0][1], "répétitions")
        #Cherche un octet non utilisé
        while debut <256 and LUToctetsdispo[debut] == False:
            debut += 1
        if debut < 256:     
            #On substitut
            imageout = imageout.replace(paires[0][0], chr(debut))
            LUToctetsdispo[debut] = False
            dictsymb += [[paires[0][0], chr(debut)]]
        else:
            print("Il n'y a plus d'octets disponible!") #Bien sûr, ce n'est pas exact car la recherche commence à imageout[0]
        
        print(imageout)
        print(dictsymb)
    else:
        remplacementpossible = False


temps_execution = time.time() - start_time
print("Temps d execution : ", time.time() - start_time, " secondes")

print("Longueur = {0}".format(np.ceil(np.log2(nbsymboles))*len(imageout)))
print("Longueur originale = {0}".format(longueurOriginale))

