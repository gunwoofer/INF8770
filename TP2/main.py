import matplotlib.pyplot as py
import numpy as np
import scipy.fftpack as dctpack
from anytree import Node, RenderTree, PreOrderIter, AsciiStyle
INDEX_Y = 0
INDEX_CB = 1
INDEX_CR = 2

INDEX_R = 0
INDEX_G = 1
INDEX_B = 2

BLOCK_SIZE = 8

Quant = np.matrix('1 1 1 1 1 1 1 1;\
1 1 1 1 1 1 1 1;\
1 1 1 1 1 1 1 1;\
1 1 1 1 1 1 1 1;\
1 1 1 1 1 1 1 1;\
1 1 1 1 1 1 1 1;\
1 1 1 1 1 1 1 1;\
1 1 1 1 1 1 1 1').astype('float')

print ("TP2 INF8770..")

image = py.imread("ImageFinal4.jpg")
image = np.array(image)
image = image.astype('float64')

# https://stackoverflow.com/questions/34913005/color-space-mapping-ycbcr-to-rgb
def rgb2ycbcr(im):
    xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    ycbcr = im.dot(xform.T)
    ycbcr[:,:,[1,2]] += 128
    chrominance(ycbcr)
    return np.uint8(ycbcr)

def chrominance(im):
    for row in range(im.shape[0]):
        for col in range(im.shape[1]):
            oldi = row
            oldj = col
            if (row % 2 == 1):
                oldi = row - 1
            if (col % 2 == 1):
                oldj = col - 1
            im[row][col][INDEX_CB] = im[oldi][oldj][INDEX_CB]
            im[row][col][INDEX_CR] = im[oldi][oldj][INDEX_CR]

# https://stackoverflow.com/questions/34913005/color-space-mapping-ycbcr-to-rgb
def ycbcr2rgb(im):
    xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    rgb = im.astype(np.float)
    rgb[:,:,[1,2]] -= 128
    rgb = rgb.dot(xform.T)
    np.putmask(rgb, rgb > 255, 255)
    np.putmask(rgb, rgb < 0, 0)
    return np.uint8(rgb)

def diff(image, image2):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if (image[i][j] != image2[i][j]).any():
                print( "L'index ({},{}) est différent. Image1 : {}, image2 : {}".format(i, j, image[i][j], image2[i][j]))

def division8x8(image):
    bloc8x8 = []
    for row in range(0,image.shape[0] - BLOCK_SIZE + 1, BLOCK_SIZE):
        for column in range(0,image.shape[1] - BLOCK_SIZE + 1, BLOCK_SIZE):
            bloc8x8.append(image[row:row+BLOCK_SIZE,column:column+BLOCK_SIZE])
    return np.array(bloc8x8)

def inverseDivision8x8(bloc, result):
    indexbloc = 0
    for i in range(0, result.shape[0] - BLOCK_SIZE + 1, BLOCK_SIZE):
        for j in range(0, result.shape[1] - BLOCK_SIZE + 1, BLOCK_SIZE):
            result[i:i+BLOCK_SIZE,j:j+BLOCK_SIZE] = bloc[indexbloc]
            indexbloc = indexbloc + 1
    return result

def dct(image):
    img = np.zeros_like(image)
    image[:] -= 128
    for bloc in image:
        BlocDCT = dctpack.dct(dctpack.dct(bloc[:, :, 0], axis=0, norm='ortho'), axis=1, norm='ortho')
        BlocDCT2 = dctpack.dct(dctpack.dct(bloc[:, :, 1], axis=0, norm='ortho'), axis=1, norm='ortho')
        BlocDCT3 = dctpack.dct(dctpack.dct(bloc[:, :, 2], axis=0, norm='ortho'), axis=1, norm='ortho')
        bloc[:, :, 0] = BlocDCT
        bloc[:, :, 1] = BlocDCT2
        bloc[:, :, 2] = BlocDCT3 
    return image

def idct(image):
    img = np.zeros_like(image)
    for bloc in image:
        BlocIDCT = dctpack.idct(dctpack.idct(bloc[:, :, 0], axis=0, norm='ortho'), axis=1, norm='ortho')
        BlocIDCT2 = dctpack.idct(dctpack.idct(bloc[:, :, 1], axis=0, norm='ortho'), axis=1, norm='ortho')
        BlocIDCT3 = dctpack.idct(dctpack.idct(bloc[:, :, 2], axis=0, norm='ortho'), axis=1, norm='ortho')
        bloc[:, :, 0] = BlocIDCT
        bloc[:, :, 1] = BlocIDCT2
        bloc[:, :, 2] = BlocIDCT3 
    image[:] += 128
    return image

def quantification(image):
    for bloc in image:
        bloc[:, :, 0] = np.round(np.divide(bloc[:, :, 0], Quant))
        bloc[:, :, 1] = np.round(np.divide(bloc[:, :, 1], Quant))
        bloc[:, :, 2] = np.round(np.divide(bloc[:, :, 2], Quant))
    return image


def dequantification(image):
    for bloc in image:
        bloc[:, :, 0] = np.round(np.multiply(bloc[:, :, 0], Quant))
        bloc[:, :, 1] = np.round(np.multiply(bloc[:, :, 1], Quant))
        bloc[:, :, 2] = np.round(np.multiply(bloc[:, :, 2], Quant))
    return image

# https://medium.com/100-days-of-algorithms/day-63-zig-zag-51a41127f31?fbclid=IwAR2aFNqEVXFlROq9GJ9wJ7L3ura1M-EHb8666E2KKrZ9Sne30-xfVKByxjY
def zig_zag_index(k, n):
    # upper side of interval
    if k >= n * (n + 1) // 2:
        i, j = zig_zag_index(n * n - 1 - k, n)
        return n - 1 - i, n - 1 - j
    # lower side of interval
    i = int((np.sqrt(1 + 8 * k) - 1) / 2)
    j = k - i * (i + 1) // 2
    return (j, i - j) if i & 1 else (i - j, j)


def zigzag(image):
    result = []
    for bloc in image:
        array = np.zeros((BLOCK_SIZE*BLOCK_SIZE, 3))
        for k in range (BLOCK_SIZE * BLOCK_SIZE):
            t = zig_zag_index(k, BLOCK_SIZE)
            array[k] = bloc[t]
        array.flatten()
        result.append(array)
    return result
       
def inverseZigZag(image):
    result = []
    for bloc in image:
        M = np.zeros((BLOCK_SIZE, BLOCK_SIZE, 3), dtype=float)
        for k in range (BLOCK_SIZE * BLOCK_SIZE):
            t = zig_zag_index(k, BLOCK_SIZE)
            M[t] = bloc[k]
        result.append(M)  
    return result


def RLE(image):
    result = []
    for blocs in image:
        resultBloc = []
        for ycbcr in blocs:
            retirer0negatif(ycbcr)
        M = concatenateYCbCr(blocs)
        i = 0
        for elem in M:
            if elem == '0.0,0.0,0.0':
                i += 1
            else:
                resultBloc.append("({}, {}".format(elem, i))
                i = 0
        result.append(resultBloc)
    return result

def inverseRLE(image):
    M = np.zeros_like(image) # image est un tableau, pas un np array. Il 
    for bloc in image: 
        i = 0
        k = 0
        temp = []
        for elem in bloc:
            Y, cb, cr, nb0 = elem.replace('(','').replace(')', '').split(',')
            j = 0
            while j < int(nb0):
                M[i + k] = (0.0, 0.0, 0.0)
                k += 1
                j += 1
            M[i + k] = (float(Y), float(cb), float(cr)) #ERREUR contient tuple au lieu d'array 64
            i += 1
    return M

def retirer0negatif(elem):
    if elem == '-0.0':
        return '0.0'
    else:
        return elem

def concatenateYCbCr(bloc):
    M = []
    for k in range(BLOCK_SIZE * BLOCK_SIZE):
        a = retirer0negatif(str(bloc[k][0]))
        b = retirer0negatif(str(bloc[k][1]))
        c = retirer0negatif(str(bloc[k][2]))
        M.append("{},{},{}".format(a,b,c))
    return M


def huffman(image):
    Message = ''.join(image)

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
    MessageCode = []
    longueur = 0 
    for i in range(len(Message)):
        substitution = list(filter(lambda x: x[0] == Message[i], dictionnaire))
        MessageCode += [substitution[0][1]]
        longueur += len(substitution[0][1])

    return (MessageCode, dictionnaire)



# https://stackoverflow.com/questions/33089660/decoding-a-huffman-code-with-a-dictionary
def huffmanDecode (dictionary, text):
    res = ""
    text = ''.join(text)
    while text:
        for k in dictionary:
            if text.startswith(k[1]):
                res += k[0]
                text = text[len(k[1]):]
    return res

print("rgb -> YCbCr..")
image = rgb2ycbcr(image)

print("Division en blocs de 8..")
imagebloc = division8x8(image)
imagebloc = imagebloc.astype('float64')

print("DCT..")
imagedct = dct(imagebloc)

print('quantification..')
imageQuantifie = quantification(imagedct)

print("zigzag..")
imageZigzag = zigzag(imageQuantifie)

print('RLE')
imageRLE = RLE(imageZigzag)

print('Huffman')

messages = []
dictionnaires = []
for bloc in imageRLE:
    message, dictionnaire = huffman(bloc)
    messages.append(message)
    dictionnaires.append(dictionnaire)

size = 0
for message in messages: 
    size += len(message)
print(size)

print("INVERSION")


print('Inverse Huffman')
blocs = []
for i in range(0, len(messages)):
    blocs.append(huffmanDecode(dictionnaires[i], messages[i]))

print('inverse RLE')
imageInverseRLE = inverseRLE(imageRLE)


print("InverseZigZag")
imageInverseZigZag = inverseZigZag(imageZigzag) # Inversion imageZigZag imageInverseRLE
imageInverseZigZag = np.array(imageInverseZigZag)
print('dequantification..')
imageDequantifie = dequantification(imageInverseZigZag)


print('IDCT ...')
imageidct = idct(imageDequantifie)

print("Reconstruction des blocs de 8..")
image = inverseDivision8x8(imageidct, image)

print("YCbCr -> rgb..")
image = ycbcr2rgb(image)

image = image.astype('uint8')

py.imshow(image)
py.show()