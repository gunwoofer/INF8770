import matplotlib.pyplot as plt 
  

# En continu 

# x axis values 
x = [2**1, 2**2, 2**3, 2**4, 2**5, 2**6, 2**7, 2**8, 2**9, 2**10, 2**11, 2**12, 2**13, 2**14, 2**15, 2**16, 2**17, 2**18, 2**19, 2**20] 
# corresponding y axis values 
y_pair_octet = [0.00004697, 0.000072, 0.00015, 0.00038, 0.0007, 0.00119, 0.0017, 0.003, 0.0058, 0.012, 0.02379, 0.068, 0.12346, 0.2379, 0.4, 0.97, 1.74, 3.27, 6.54, 15.02] 
y_predictif = [1.79, 1.83, 1.77, 1.867, 1.75, 1.78, 1.77, 1.78, 1.79, 1.80, 1.87, 1.81, 1.86, 2.03, 2.23, 2.75, 4.3, 5.86, 14.95, 17.54]  

y_pair_octet_alterner = [0.0001, 0.00009, 0.00016, 0.00028, 0.0005, 0.0013, 0.0023, 0.004, 0.008, 0.016, 0.033, 0.07, 0.16, 0.30, 0.66, 1.24, 2.50, 4.958, 10.24, 20.17] 
y_predictif_alterner = [0.59, 0.55, 0.54, 0.55, 0.56, 0.57, 0.58, 0.58, 0.59, 0.62, 0.67, 0.67, 0.79, 1.004, 1.44, 2.38, 4.19, 8.056, 15.37, 30] 

# plotting the points  
plt.plot(x, y_predictif, label="predictif + huffman continu") 
plt.plot(x, y_pair_octet, label="pair d'octet continu") 
plt.plot(x, y_pair_octet_alterner, label="pair d'octet alterner") 
plt.plot(x, y_predictif_alterner, label="predictif alterner") 

# naming the x axis 
plt.xlabel('Taille initial') 
# naming the y axis 
plt.ylabel("Temps d'execution") 
  
# giving a title to my graph 
plt.title("Evolution du temps d'execution") 
  
plt.legend() 

# function to show the plot 
plt.show() 

# Par Longueur

x = [2**1, 2**2, 2**3, 2**4, 2**5, 2**6, 2**7, 2**8, 2**9, 2**10, 2**11, 2**12, 2**13, 2**14, 2**15, 2**16, 2**17, 2**18, 2**19, 2**20] 
# corresponding y axis values 
y_pair_octet = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 13, 15, 17, 18, 19, 20, 21, 22] 
y_predictif = [2**1, 2**2, 2**3, 2**4, 2**5, 2**6, 2**7, 2**8, 2**9, 2**10, 2**11, 2**12, 2**13, 2**14, 2**15, 2**16, 2**17, 2**18, 2**19, 2**20]  

y_pair_octet_alterner = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]  
y_predictif_alterner = [2**1, 2**2, 2**3, 2**4, 2**5, 2**6, 2**7, 2**8, 2**9, 2**10, 2**11, 2**12, 2**13, 2**14, 2**15, 2**16, 2**17, 2**18, 2**19, 2**20]  

# plotting the points  
#plt.plot(x, y_predictif, label="predictif + huffman continu") 
plt.plot(x, y_pair_octet, label="pair d'octet continu") 
plt.plot(x, y_pair_octet_alterner, label="pair d'octet alterner") 
#plt.plot(x, y_predictif_alterner, label="predictif alterner") 

# naming the x axis 
plt.xlabel('Taille initial') 
# naming the y axis 
plt.ylabel("Temps d'execution") 
  
# giving a title to my graph 
plt.title("Evolution de la longueur du message") 
  
plt.legend() 

# function to show the plot 
plt.show() 