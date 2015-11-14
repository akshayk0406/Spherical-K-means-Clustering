import matplotlib.pyplot as plt
import numpy as np
import sys

entropy_file = sys.argv[1]
output_file = sys.argv[2]
i = 1
x = []
y = []
intensity = []
with open(entropy_file,'r') as f:
	for line in f:
		intensity.append([int(freq.strip()) for freq in line.split(" ")[:-1]])
		x.append(i)
		y.append(i*0.1)
		i = i+1

#setup the 2D grid with Numpy
x, y = np.meshgrid(x, y)

#convert intensity (list of lists) to a numpy array for plotting
intensity = np.array(intensity)

#now just plug the data into pcolormesh, it's that easy!
plt.pcolormesh(x, y, intensity)
plt.colorbar() #need a colorbar to show the intensity scale
plt.savefig(output_file)
