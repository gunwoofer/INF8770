 # -*-coding:Latin-1 -* 

import numpy as np
import matplotlib.pyplot as py
import time
from anytree import Node, RenderTree, PreOrderIter, AsciiStyle
from guppy import hpy

h = hpy()
print ("Codage Predictif")

def rgb2gray(rgb):
    return np.dot(rgb[:,:], [0.299, 0.587, 0.114])

fig1 = py.figure(figsize = (10,10))
imagelue = py.imread('degrade.jpg')
image=imagelue.astype('float')
image=rgb2gray(image)
imageout=image.astype('uint8')
py.imshow(imageout,cmap = py.get_cmap('gray'))
#py.show()

#hist, intervalles = np.histogram(imageout, bins=256)
#py.bar(intervalles[:-1], hist, width = 2)
#py.xlim(min(intervalles)-1, max(intervalles))
#py.show()

#image = "001100011110001010101011110001101010111111111111110000000000000000110101010100011101010101"

#image =""
#for i in range(32761):
#    if i < 32761 / 2:
#       image += '0'
#    else:
#        image += '1'
#image = list(image)
#image = np.reshape(image,(-1,181))
#image = image.astype('uint8')
#imageout=image.astype('uint8')
#py.imshow(imageout,cmap = py.get_cmap('gray'))
#py.show()
# Debut du Timer
start_time = time.time()

col=image[:,0]
image = np.column_stack((col,image))
col=image[:,len(image[0])-1]
image = np.column_stack((col,image))
row=image[0,:]
image = np.row_stack((row,image))
row=image[len(image)-1,:]
image = np.row_stack((row,image))

matpred = [[0.33,0.33],[0.33,0.0]]


erreur = np.zeros((len(image)-2,len(image[0])-2))
imagepred = np.zeros((len(image)-2,len(image[0])-2))
for i in range(1,len(image)-2):
    for j in range(1,len(image[0])-2):
        imagepred[i][j]=image[i-1][j-1]*matpred[0][0]+image[i-1][j]*matpred[0][1]+image[i][j-1]*matpred[1][0]
        erreur[i][j]=imagepred[i][j]-image[i][j]

temps_execution = time.time() - start_time
print("Temps d execution : ", time.time() - start_time, " secondes")

hist, intervalles = np.histogram(erreur, bins=100)
py.bar(intervalles[:-1], hist, width = 2)
py.xlim(min(intervalles)-1, max(intervalles))
#py.show()

fig2 = py.figure(figsize = (10,10))
imageout=imagepred.astype('uint8')
py.imshow(imageout, cmap = py.get_cmap('gray'))
#py.show()


fig3 = py.figure(figsize = (10,10))
erreur=abs(erreur)*5
imageout=erreur.astype('uint8')
py.imshow(imageout, cmap = py.get_cmap('gray'))
imageout = imageout.flatten().astype('str')


Message = ''.join(imageout)

ArbreSymb =[[Message[0], Message.count(Message[0]), Node(Message[0])]] 
#dictionnaire obtenu à partir de l'arbre.
dictionnaire = [[Message[0], '']]
nbsymboles = 1

#Recherche des feuilles de l'arbre

for i in range(1,len(Message)):
    if not list(filter(lambda x: x[0] == Message[i], ArbreSymb)):
        ArbreSymb += [[Message[i], Message.count(Message[i]),Node(Message[i])]]
        dictionnaire += [[Message[i], '']]
        nbsymboles += 1
        
longueurOriginale = np.ceil(np.log2(nbsymboles))*len(Message)

ArbreSymb = sorted(ArbreSymb, key=lambda x: x[1])
#print(ArbreSymb)



while len(ArbreSymb) > 1:
    #Fusion des noeuds de poids plus faibles
    symbfusionnes = ArbreSymb[0][0] + ArbreSymb[1][0] 
    #Création d'un nouveau noeud
    noeud = Node(symbfusionnes)
    temp = [symbfusionnes, ArbreSymb[0][1] + ArbreSymb[1][1], noeud]
    #Ajustement de l'arbre pour connecter le nouveau avec ses parents 
    ArbreSymb[0][2].parent = noeud
    ArbreSymb[1][2].parent = noeud
    #Enlève les noeuds fusionnés de la liste de noeud à fusionner.
    del ArbreSymb[0:2]
    #Ajout du nouveau noeud à la liste et tri.
    ArbreSymb += [temp]
    #Pour affichage de l'arbre ou des sous-branches
   # print('\nArbre actuel:\n\n')
    for i in range(len(ArbreSymb)):
        if len(ArbreSymb[i][0]) > 1:
            #print(RenderTree(ArbreSymb[i][2], style=AsciiStyle()).by_attr())   
            pass
    ArbreSymb = sorted(ArbreSymb, key=lambda x: x[1])  
    #print(ArbreSymb)


ArbreCodes = Node('')
noeud = ArbreCodes
#print([node.name for node in PreOrderIter(ArbreSymb[0][2])])
parcoursprefix = [node for node in PreOrderIter(ArbreSymb[0][2])]
parcoursprefix = parcoursprefix[1:len(parcoursprefix)] #ignore la racine

Prevdepth = 0 #pour suivre les mouvements en profondeur dans l'arbre
for node in parcoursprefix:  #Liste des noeuds 
    if Prevdepth < node.depth: #On va plus profond dans l'arbre, on met un 0
        temp = Node(noeud.name + '0')
        noeud.children = [temp]
        if node.children: #On avance le "pointeur" noeud si le noeud ajouté a des enfants.
            noeud = temp
    elif Prevdepth == node.depth: #Même profondeur, autre feuille, on met un 1
        temp = Node(noeud.name + '1')
        noeud.children = [noeud.children[0], temp]  #Ajoute le deuxième enfant
        if node.children: #On avance le "pointeur" noeud si le noeud ajouté a des enfants.
            noeud = temp
    else:
        for i in range(Prevdepth-node.depth): #On prend une autre branche, donc on met un 1
            noeud = noeud.parent #On remontre dans l'arbre pour prendre la prochaine branche non explorée.
        temp = Node(noeud.name + '1')
        noeud.children = [noeud.children[0], temp]
        if node.children:
            noeud = temp
        
    Prevdepth = node.depth    
    
#print('\nArbre des codes:\n\n',RenderTree(ArbreCodes, style=AsciiStyle()).by_attr())         
#print('\nArbre des symboles:\n\n', RenderTree(ArbreSymb[0][2], style=AsciiStyle()).by_attr())

ArbreSymbList = [node for node in PreOrderIter(ArbreSymb[0][2])]
ArbreCodeList = [node for node in PreOrderIter(ArbreCodes)]

for i in range(len(ArbreSymbList)):
    if ArbreSymbList[i].is_leaf: #Génère des codes pour les feuilles seulement
        temp = list(filter(lambda x: x[0] == ArbreSymbList[i].name, dictionnaire))
        if temp:
            indice = dictionnaire.index(temp[0])
            dictionnaire[indice][1] = ArbreCodeList[i].name
            
#print(dictionnaire)


MessageCode = []
longueur = 0 
for i in range(len(Message)):
    substitution = list(filter(lambda x: x[0] == Message[i], dictionnaire))
    MessageCode += [substitution[0][1]]
    longueur += len(substitution[0][1])

#print(MessageCode)
print(h.heap())
print("Temps d execution : ", time.time() - start_time, " secondes")

print("Longueur = {0}".format(longueur))
print("Longueur originale = {0}".format(longueurOriginale))