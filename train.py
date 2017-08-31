import os
import os.path
import argparse
import h5py
from random import randrange
import numpy as np
from sklearn.externals import joblib
import pickle
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model, datasets
import matplotlib.pyplot as plt
from sklearn import tree


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type = str  )
parser.add_argument("--weights_path", type = str)
parser.add_argument("--train_data", type = str  )
parser.add_argument("--plots_save_dir", type = str  )

args = parser.parse_args()
# args.model_name='LogisticRegression'
# args.weights_path = 'Weights/'
# args.train_data = 'part_B_train.h5'
# args.plots_save_dir = 'Plots/'

# Load the test data
def load_h5py(filename):	
	with h5py.File(filename, 'r') as hf:
		X = hf['X'][:]
		Y = hf['Y'][:]
	return X, Y
def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)
def cross_validation_split(datasetX,datasetY, folds=5):
	datasetX_split = []
	datasetY_split = []
	fold_size = int(len(datasetX)/folds)
	idx = 0
	for i in range(folds):
		foldX = list()
		foldY = []
		for j in range(idx,idx+fold_size):
			foldX.append(datasetX[j])
			foldY.append(datasetY[j])
		idx = idx+fold_size
		datasetX_split.append(foldX)
		datasetY_split.append(foldY)
	return datasetX_split, datasetY_split



filename = args.train_data	
X, Y = load_h5py(filename)

Y_ = []
for y in Y:
	for i, r in enumerate(y):
		if r==1.0:
			Y_.append(i)
Y = Y_
num_folds = 5
resX, resY = cross_validation_split(X,Y,num_folds)


if args.model_name == 'GaussianNB':
	scores=[]
	model = GaussianNB()
	for i in range(num_folds):
		testX = resX[i]
		trainX =[]
		testY = resY[i]
		trainY =[]
		for k in range(num_folds):
			if(k!=i):
				trainX.extend(resX[k])
				trainY.extend(resY[k])
		trainX = np.array(trainX)
		trainY = np.array(trainY)
		model.fit(trainX,trainY)
		predicted = model.predict(testX)
		print predicted
		# print "*********************************************************************"
		# print testY
		print "Accuracy for fold %d as test"%(i+1)
		score = accuracy_score(testY,predicted)
		scores.append(score)
		print score
		print ""
		print ""
		print "______________________________________________________________________"
	print "ACCURACY OF GAUSSIANNB : %f"%(mean(scores))
	joblib.dump(model,args.weights_path+"Best" +args.model_name+args.train_data+'.pkl' )
	pass



elif args.model_name == 'LogisticRegression':
	dict1 = {}
	dict2 = {}
	penalty = ['l1','l2']
	C = [1.0,1.5]
	max_iter = [100,150,200]
	verbose = [0]
	scores = []
	accuracy = []
	l=0;
	models=[]
	for p in penalty:
		for c in C:
			for max_i in max_iter:
				for v in verbose:
					for i in range(num_folds):
						testX = resX[i]
						trainX =[]
						testY = resY[i]
						trainY =[]
						for k in range(num_folds):
							if(k!=i):
								trainX.extend(resX[k])
								trainY.extend(resY[k])
						trainX = np.array(trainX)
						trainY = np.array(trainY)
						logreg = linear_model.LogisticRegression(verbose=v, penalty=p,C=c,max_iter=max_i,solver='saga')
						models.append(logreg)
						r = logreg.fit(trainX,trainY)
						Z = logreg.predict(testX)
						score = accuracy_score(testY,Z)
						scores.append(score)
					temp = []
					dict1[l] = mean(scores)

					temp.append(p)
					temp.append(c)
					temp.append(max_i)
					temp.append(v)

					dict2[l] = temp
					l=l+1
					accuracy.append(mean(scores))
	maximum = max(dict1, key=dict1.get)
	par = dict2[maximum]
	logreg = models[maximum]
	joblib.dump(logreg,args.weights_path+"Best" +args.model_name+args.train_data+'.pkl' )
	xTicks = dict2.values()
	x = np.arange(len(dict2))
	y = dict1.values()
	print y
	plt.ylim(.7, 1.0)
	plt.bar(x,y,align = 'center',alpha=0.5)
	plt.xticks(x,xTicks,rotation=45)
	plt.ylabel("Accuracy")
	plt.xlabel("Parameters")
	plt.title("Accuracy vs Parameters")
	string = args.train_data + args.model_name +'graph.png'
	my_file = string
	plt.savefig(args.plots_save_dir+string)
	plt.show()
	
	pass



elif args.model_name == 'DecisionTreeClassifier':
	# define the grid here

	# do the grid search with k fold cross validation

	# model = DecisionTreeClassifier(  ...  )

	# save the best model and print the result
	max_depth = [None,1]
	min_samples_split = [2,4]
	min_samples_leaf = [1,2]
	max_features = [None,"auto","log2"]

	dict1 = {}
	dict2 = {}

	l=0
	models=[]
	#keep one as test set and others as train set
	scores = []
	accuracy = []
	for d in max_depth:
		for s in min_samples_split:
			for sl in min_samples_leaf:
				for f in max_features:
					# print "__________________________________________________________________________________"
					for i in range(num_folds):
						# print "FOLD NUMBER %d"%i
						testX = resX[i]
						trainX =[]
						testY = resY[i]
						trainY =[]
						for k in range(num_folds):

							if(k!=i):
								trainX.extend(resX[k])
								trainY.extend(resY[k])
						trainX = np.array(trainX)
						trainY = np.array(trainY)
						clf = tree.DecisionTreeClassifier(max_depth=d,min_samples_leaf=sl,min_samples_split=s,max_features=f)
						models.append(clf)
						clf = clf.fit(trainX, trainY)
						Z = clf.predict(testX)
						score = accuracy_score(testY,Z)
						# print score
						scores.append(score)
					temp = []
					dict1[l] = mean(scores)

					temp.append(d)
					temp.append(s)
					temp.append(sl)
					temp.append(f)

					dict2[l] = temp
					l=l+1
					accuracy.append(mean(scores))
	print "******************************************************************"
	# print dict1
	#print dict2
	maximum = max(dict1, key=dict1.get)
	par = dict2[maximum]
	print maximum
	clf = models[maximum]
	joblib.dump(clf,args.weights_path+"Best" +args.model_name+args.train_data+'.pkl' )
	xTicks = dict2.values()
	x = np.arange(len(dict2))
	y = dict1.values()
	print y
	plt.ylim(.4, 1.0)
	plt.bar(x,y,align = 'center',alpha=0.5)
	plt.xticks(x,xTicks,rotation=90)
	plt.ylabel("Accuracy")
	plt.xlabel("Parameters")
	plt.title("Accuracy vs Parameters")
	string = args.train_data + args.model_name +'graph.png'
	my_file = string
	plt.savefig(args.plots_save_dir+string)
	plt.show()	


	pass
else:
        raise Exception("Invald Model name")
