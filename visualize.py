import os
import os.path
import argparse
from sklearn.manifold import TSNE
import h5py
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from sklearn import (manifold, datasets, decomposition, ensemble,discriminant_analysis, random_projection)
from time import time

parser = argparse.ArgumentParser()
parser.add_argument("--data", type = str  )
parser.add_argument("--plots_save_dir", type = str  )
args = parser.parse_args()

filename1 = args.data	
f1 = h5py.File(filename1,'r')

X_grp_key = f1.keys()[0]
Y_grp_key = f1.keys()[1]
datasets
X = f1[X_grp_key]
Y = f1[Y_grp_key]
classes= Y.shape[1]
Y_ = []
for y in Y:
	for i, r in enumerate(y):
		if r==1.0:
			Y_.append(i)
X = np.array(X)
Y = np.array(Y)
Y_ = []
for y in Y:
	for i, r in enumerate(y):
		if r==1.0:
			Y_.append(i)
			
tsne = manifold.TSNE()
print "transform"
t0 = time() 
X_tsne = tsne.fit_transform(X)
print "transformed"
print time()-t0

color_map = {0:'red', 1:'blue', 2:'lightgreen', 3:'purple', 4:'cyan',5:'black',6:'brown',7:'pink',8:'yellow',9:'orange'}
fig=plt.figure()
for i in range(min(len(color_map),classes)):
	# X_tsne.shape[0]
	x = []
	y = []
	for j in range(X_tsne.shape[0]):
		if(color_map[i] == color_map[Y_[j]]):
			x.append(X_tsne[j,0])
			y.append(X_tsne[j,1])
	plt.scatter(x=x,y=y,c=color_map[i],label=i)
    # plt.scatter(x=X_tsne[i,0], y=X_tsne[i,1], c=color_map[Y_[i]])

plt.legend(loc=1)
plt.xlabel('X in t-SNE')
plt.ylabel('Y in t-SNE')
plt.title('t-SNE visualization of test data')
string = args.data + 'graph.png'
my_file = string
plt.savefig(args.plots_save_dir+string)
plt.show()



