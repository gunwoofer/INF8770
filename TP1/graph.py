import matplotlib.pyplot as plt 
  
# x axis values 
x = [2**1, 2**2, 2**3, 2**4, 2**5, 2**6, 2**7, 2**8, 2**9, 2**10, 2**11, 2**12, 2**13, 2**14, 2**15, 2**16, 2**17, 2**18, 2**19, 2**20] 
# corresponding y axis values 
y_pair_octet = [0.00004697, 0.000072, 0.00015, 0.00038, 0.0007, 0.00119, 0.0017, 0.003, 0.0058, 0.012, 0.02379, 0.068, 0.12346, 0.2379, 0.4, 0.97, 1.74, 3.27, 6.54, 15.02] 
y_predictif = [0.005, 0.60958, 6.088, 31.537, 64.883]  
# plotting the points  
#plt.plot(x, y_predictif, label="predictif + huffman") 
plt.plot(x, y_pair_octet, label="pair d'octet") 

# naming the x axis 
plt.xlabel('Taille initial') 
# naming the y axis 
plt.ylabel("Temps d'execution") 
  
# giving a title to my graph 
plt.title("Evolution du temps d'execution avec une echelle logarithmique (log2)") 
  
plt.legend() 

# function to show the plot 
plt.show() 