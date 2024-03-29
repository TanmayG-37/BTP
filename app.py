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











# from scipy.special import erfc

class Chauv:
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
		self.size=len(data[0])
		print(self.size)
		# get_bins gives us the factors for each feature
		self.get_bins()
		# clusters functions hashes the datapoints into groups so that we can detect local outliers
		self.clusters(pd.DataFrame(data))
		self.mean=data.mean(axis=0)
		self.std=data.std(axis=0)+(10**-5) # to avoid division by 0 error
		return None


	def chauvenet(self,data):
		N = len(data)                
		criterion = 1.0/(2*N)         # Chauvenet's criterion
		d = abs(data-self.mean)/self.std         
		d /= 2.0**0.5                
		# prob = erfc(d)               # Area normal dist.    
		print(prob)
		filter = prob >= criterion 
		# print(filter)  
		self.scores = prob
		return filter                




class Robust:
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
		self.size=len(data[0])
		print(self.size)
		# get_bins gives us the factors for each feature
		self.get_bins()
		# clusters functions hashes the datapoints into groups so that we can detect local outliers
		self.clusters(pd.DataFrame(data))
		self.med = pd.DataFrame(data).median(axis=0)
		med = np.median(data)
		# x   = abs(data-med)
		# MAD = np.median(x)

		self.mad = np.mad(data)
		return None



	def score(self,data):
		# improve calculation of score using numpy arrays for matrix multiplication
		# # 1. scoring on the based of standard deviation (regressive annd a lot of calculations)
		temp_data=(0.6745*(pd.DataFrame(data)-self.med))/self.mad
		for i in range(self.L):
			self.scores.append((np.dot(temp_data,self.weights[i].transpose()))/self.size) 
		return self.scores











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



@app.route('/input', methods=['GET', 'POST'])
def take_input():
	return render_template('base3.html')



rno = ''
@app.route('/data', methods=['GET', 'POST'])
def data():
	global f
	global rno
	import time
	start_time = time.time()

	if request.method == 'POST':
		f = request.files['csvfile']
		algo = request.form['algo']
		if not os.path.isdir('static'):
			os.mkdir('static')
		filepath = os.path.join('static', f.filename)
		f.save(filepath)	
		

	# Reading the data
	data = pd.read_csv('static/'+str(f.filename))
	# data = data.astype('float')
	# data = data.dropna(axis='columns')




	# Data Analytics
	number = len(data)
	features = len(data.iloc[0])











	out = []
	# 1st method - Random Z score 

	print(algo)


	if algo == 'RZScore':
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
		out = []
		t1 = q3+(1.5)*iqr; t2 = q1-(1.5)*iqr
		for i in range(len(new)):
			if new[i]>t1 or new[i]<t2:
				out.append(1)
			else:
				out.append(0)


		# extreme = []
		# for i in range(len(new)):
		# 	if new[i]>(q3+(3)*iqr) or new[i]<(q1-(3)*iqr):
		# 		extreme.append(i)
		# else:
		# 	extreme.append(0)
		




	# 2nd method - isolation forest
	elif algo == 'Isolation Forest':
		obj = IF(50)
		obj.partition(data)
		temp_scores = obj.score_samples(data)
		out = obj.labels


	# 3rd method HBOS
	elif algo=='HBOS':
		from pyod.models import hbos
		model = hbos.HBOS()
		model.fit(data)
		out = model.predict(data)
		temp_scores = model.decision_function(data)


	#4th method Robust
	elif algo=='Robust RZ':
		obj=Robust(50)
		obj.fit(data)
		obj.score(data)
		temp_scores = obj.scores
		new = np.array([0.0 for i in range(len(obj.scores[0]))])
		for i in range(len(temp_scores)):
			new+=temp_scores[i]

		
		q1 = np.quantile(new,.25)
		q3 = np.quantile(new,.75)
		iqr = q3-q1
		out = []
		t1 = q3+(1.5)*iqr; t2 = q1-(1.5)*iqr
		for i in range(len(new)):
			if new[i]>t1 or new[i]<t2:
				out.append(1)
			else:
				out.append(0)


	# 5th method
	else:
		obj=Chauv(50)
		obj.fit(data)
		obj.score(data)
		temp_scores = obj.scores
		new = np.array([0.0 for i in range(len(obj.scores[0]))])
		for i in range(len(temp_scores)):
			new+=temp_scores[i]

		
		q1 = np.quantile(new,.25)
		q3 = np.quantile(new,.75)
		iqr = q3-q1
		out = []
		t1 = q3+(1.5)*iqr; t2 = q1-(1.5)*iqr
		for i in range(len(new)):
			if new[i]>t1 or new[i]<t2:
				out.append(1)
			else:
				out.append(0)







	# We declare a point as an outlier if it occurs in 2 or more methods
	output = 0
	# labels = out
	for i in range(len(out)):
		if out[i]==1:
			output+=1

	
	# Section fopr making visual representations
	# import matplotlib.pyplot as plt
	# from sklearn.manifold import TSNE
	# from sklearn.decomposition import PCA
	# temp_data = PCA(n_components=2).fit_transform(data)
	# temp_data = pd.DataFrame(temp_data)
	# temp_data['labels'] = labels
	# data['RZScore_HBOS_IF_Outlier_s'] = labels
	# t0 = temp_data[temp_data['labels']==0]
	# t0.drop(columns=['labels'], inplace=True)
	# t1 = temp_data[temp_data['labels']==1]
	# t1.drop(columns=['labels'], inplace=True)


	# plt.scatter(t0[0],t0[1], color='red', s=0.3)
	# plt.scatter(t1[0],t1[1], color='blue', s=0.75)
	# plt.xlabel('F1')
	# plt.ylabel('F2')
	# rno = str(random.randint(0, 1000000))
	# plt.savefig('static/graph_photo/tsne'+rno+'.jpg')
	# full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'tsne'+rno+'.jpg')
	# Image.open('static/graph_photo/tsne.png').save('static/graph_photo/tsne.jpg','JPEG')


	# impurity
	impurity = (output*100)/number



	# saving outliers indices
	temp_dict = {'Outlier':out}
	df = pd.DataFrame(temp_dict)
	rno = str(random.randint(0, 1000000))
	df.to_csv('static/outlier'+rno+'.csv')



	# score analysis
	temp_scores = temp_scores + min(temp_scores)
	temp_scores = temp_scores / max(temp_scores)
	std = round(np.std(temp_scores),3)
	mean = round(np.mean(temp_scores),3)

	return render_template('base2.html', outliers=output, number=number, features=features, impurity=impurity, std=std, mean=mean)



@app.route('/download', methods=['GET', 'POST'])
def download():
	global rno
	print('ahahahahaha')
	return send_file('static/outlier'+rno+'.csv', as_attachment=True, attachment_filename='out.csv')



@app.route('/contact', methods=['GET', 'POST'])
def contact():
	return render_template('base4.html')








if __name__=='__main__':
	app.run(debug=True)
