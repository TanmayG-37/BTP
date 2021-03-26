import flask
import pandas as pd
import numpy as np
import copy
import pickle
import csv
import random
from PIL import Image
from sklearn.ensemble import IsolationForest

class RZScore:
	def __init__(self,L):
		self.L=L
		self.mean=[]
		self.std=[]
		self.weights=[]
		self.size=0
		self.scores=[]



	def get_bins(self,data):
		# defining the weights
		for i in range(self.L):
			self.weights.append(np.array([random.random() for j in range(self.size)])) 
			


	def fit(self,data):
		self.size=len(data.iloc[0])
		# print("---%s seconds---" % (time.time()-start_time))
		self.get_bins(pd.DataFrame(data))
		# print("---%s seconds---" % (time.time()-start_time))
		self.mean=data.mean(axis=0)
		self.std=data.std(axis=0)+(10**-5) # to avoid division by 0 error
		print(self.std)



	def score(self,data):
		# improve calculation of score using numpy arrays for matrix multiplication
		# # 1. scoring on the based of standard deviation (regressive annd a lot of calculations)
		temp_data=(data-self.mean)/self.std
		for i in range(self.L):
			self.scores.append((np.dot(temp_data,self.weights[i].transpose()))/self.size) 
		return self.scores






class IF:
	def __init__(self,L):
		self.L=L
		self.scores=[]
		self.labels = []


	def partition(self,data):
		est = IsolationForest(n_estimators=self.L, max_features=1, random_state=42)
		est.fit(data)
		self.scores = est.score_samples(data)
		self.labels = est.predict(data)
		return














##################################
from flask import *
import os
GRAPH_FOLDER = os.path.join('static', 'graph_photo')
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = GRAPH_FOLDER

@app.route('/', methods=['GET', 'POST'])
def input():
	# global f
	# if request.method == 'POST':
	# 	f = request.files['csvfile']
	# 	if not os.path.isdir('static'):
	# 		os.mkdir('static')
	# 	filepath = os.path.join('static', f.filename)
	# 	f.save(filepath)
		
	# 	# return f.filename
	return render_template('base.html')


@app.route('/data', methods=['GET', 'POST'])
def data():
	global f
	import time
	start_time = time.time()

	if request.method == 'POST':
		f = request.files['csvfile']
		if not os.path.isdir('static'):
			os.mkdir('static')
		filepath = os.path.join('static', f.filename)
		f.save(filepath)	
		

	# Reading the data
	data = pd.read_csv('static/'+str(f.filename))
	data = data.astype('float')
	data = data.dropna(axis='columns')


	# 1st method - Random Z score 
	obj=RZScore(50)
	obj.fit(data)
	obj.score(data)
	temp_scores = obj.scores
	new = np.array([0.0 for i in range(len(obj.scores[0]))])
	for i in range(len(temp_scores)):
		new+=temp_scores[i]

	
	q1 = np.quantile(new,.25)
	q3 = np.quantile(new,.75)
	iqr = q3-q1
	out1 = []
	for i in range(len(new)):
		if new[i]>(q3+(1.5)*iqr) or new[i]<(q1-(1.5)*iqr):
			out1.append(1)
		else:
			out1.append(0)


	extreme = []
	for i in range(len(new)):
		if new[i]>(q3+(3)*iqr) or new[i]<(q1-(3)*iqr):
			extreme.append(i)
		# else:
		# 	extreme.append(0)
		




	# 2nd method - isolation forest
	obj = IF(50)
	obj.partition(data)
	out2 = obj.labels


	# 3rd method HBOS
	from pyod.models import hbos
	model = hbos.HBOS()
	model.fit(data)
	out3 = model.predict(data)


	# We declare a point as an outlier if it occurs in 2 or more methods
	output = 0
	labels = []
	# types = []
	for i in range(len(out1)):
		if out1[i]==1 and (out1[i]==out2[i] or out2[i]==out3[i]):
			output+=1
			labels.append(1)
		elif out2[i]==1 and out2[i]==out3[i]:
			output+=1
			labels.append(1)
		else:
			labels.append(0)
		# if out1[i]==1 and out1[i]==out3[i]:
		# 	output+=1

	
	# Section fopr making visual representations
	import matplotlib.pyplot as plt
	from sklearn.manifold import TSNE
	temp_data = TSNE(n_components=2).fit_transform(data)
	temp_data = pd.DataFrame(temp_data)
	temp_data['labels'] = labels
	data['RZScore_HBOS_IF_Outlier_s'] = labels
	t0 = temp_data[temp_data['labels']==0]
	t0.drop(columns=['labels'], inplace=True)
	t1 = temp_data[temp_data['labels']==1]
	t1.drop(columns=['labels'], inplace=True)


	plt.scatter(t0[0],t0[1], color='red', s=0.3)
	plt.scatter(t1[0],t1[1], color='blue', s=0.75)
	plt.savefig('static/graph_photo/tsne.jpg')
	# Image.open('static/graph_photo/tsne.png').save('static/graph_photo/tsne.jpg','JPEG')

	types = []
	type1 = 0; type2 = 0; type3 = 0
	ind = []
	for i in range(len(out1)):
		if out1[i]==1 and out2[i]==1 and out3[i]==1:
			types.append(3)
			type3+=1
			ind.append(i)
		elif out1[i]==1 and (out1[i]==out2[i] or out2[i]==out3[i]):
			# output+=1
			types.append(2)
			type2+=1
		elif out2[i]==1 and out2[i]==out3[i]:
			# output+=1
			types.append(2)
			type2+=1
		elif out1[i]==1 or out2[i]==1 or out3[i]==1:
			types.append(1)
			type1+=1 
		else:
			labels.append(0)




	full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'tsne.png')
	if len(ind)==0:
		ind = 'Not Applicable'

	return render_template('base2.html', outliers=output, user_image=full_filename, t1=type1, t2=type2, t3=type3, indices=ind)







if __name__=='__main__':
	app.run(debug=True)
