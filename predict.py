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
parser.add_argument("--test_data", type = str  )
parser.add_argument("--output_preds_file", type = str  )

args = parser.parse_args()
# args.model_name='LogisticRegression'
# args.weights_path= 'Weights/BestDecisionTreeClassifierpart_B_train.h5.pkl'
# args.test_data='part_B_train.h5'
# args.output_preds_file='OUTPUT'+args.model_name+args.test_data+'.txt'

# load the test data
def load_h5py(filename):
	with h5py.File(filename, 'r') as hf:
		X = hf['X'][:]
		Y = hf['Y'][:]
	return X, Y

filename = args.test_data		
X, Y = load_h5py(filename)

Y_ = []
for y in Y:
	for i, r in enumerate(y):
		if r==1.0:
			Y_.append(i)
Y = Y_

if args.model_name == 'GaussianNB':
	filename = args.weights_path
	model = joblib.load(filename) 
	predicted = model.predict(X)
	print predicted
	file = open(args.output_preds_file,"w")
	for p in predicted:
		# u = p + "\n"
		p = str(p)
		file.write("{}\n".format(p))
	file.close()
	pass
elif args.model_name == 'LogisticRegression':
	filename = args.weights_path
	model = joblib.load(filename) 
	X=np.array(X)
	predicted = model.predict(X)
	print predicted
	file = open(args.output_preds_file,"w")
	for p in predicted:
		# u = p + "\n"
		p = str(p)
		file.write("{}\n".format(p))
	file.close()
	pass
elif args.model_name == 'DecisionTreeClassifier':
	# load the model

	# model = DecisionTreeClassifier(  ...  )
	filename = args.weights_path
	model = joblib.load(filename) 
	X=np.array(X)
	predicted = model.predict(X)
	print predicted
	file = open(args.output_preds_file,"w")
	for p in predicted:
		# u = p + "\n"
		p = str(p)
		file.write("{}\n".format(p))
	file.close()

	# save the predictions in a text file with the predicted clasdIDs , one in a new line 
	pass
else:
	raise Exception("Invald Model name")
