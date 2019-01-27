 # -*-coding:Latin-1 -* 
import numpy as np
import regex as re
import time

Message = "ABAABAABACABBABCDAADACABABAAABAABBABABAABAAB"

# Debut du Timer
start_time = time.time()

LUToctetsdispo = [True] * 256
dictsymb =[Message[0]]
LUToctetsdispo[ord(Message[0])] = False
nbsymboles = 1
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
        temppaire = Message[i]+Message[i+1]
        if not list(filter(lambda x: x[0] == temppaire, paires)): #Si la liste retournée par filter est vide.
            paires += [[temppaire,len(re.findall(temppaire, Message, overlapped = True))]]

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
            Message = Message.replace(paires[0][0], chr(debut))
            LUToctetsdispo[debut] = False
            dictsymb += [[paires[0][0], chr(debut)]]
        else:
            print("Il n'y a plus d'octets disponible!") #Bien sûr, ce n'est pas exact car la recherche commence à Message[0]
        
        print(Message)
        print(dictsymb)
    else:
        remplacementpossible = False


temps_execution = time.time() - start_time
print("Temps d execution : ", time.time() - start_time, " secondes")

print("Longueur = {0}".format(np.ceil(np.log2(nbsymboles))*len(Message)))
print("Longueur originale = {0}".format(longueurOriginale))