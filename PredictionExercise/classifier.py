import numpy as np 
from math import sqrt, pi, exp 
import random 

def gaussian_probability(obs, mu, sigma):
	num = (obs - mu) ** 2
	denom = 2 * sigma ** 2
	norm  = 1. / sqrt ( 2 * pi * sigma ** 2)
	return norm * exp( -num / denom)

class GNB(object):
	''' Gaussian Naive Bayes classifier
	'''

	def __init__(self):
		self.classes = ['left', 'keep', 'right']

	def process_vars(self, vars):
		s, d, s_dot, d_dot = vars 
		return s, d, s_dot, d_dot



	def train(self, X, Y):
		"""
		Trains the classifier with N data points and labels.

		INPUTS
		X = data: array of N observations
		  - Each observation is a tuple with 4 values: s, d, 
		    s_dot and d_dot.
		  - Example : [
			  	[3.5, 0.1, 5.9, -0.02],
			  	[8.0, -0.3, 3.0, 2.2],
			  	...
		  	]

		Y = labels: array of N labels
		  - Each label is one of "left", "keep", or "right".
		"""
		num_vars = 4

		# initialize an empty array of arrays. For this problem
		# we are looking at three labels and keeping track of 4 
		# variables for each (s,d,s_dot,d_dot), so the empty array
		# totals_by_label will look like this:

		# {
		#   "left" :[ [],[],[],[] ], 
		#   "keep" :[ [],[],[],[] ], 
		#   "right":[ [],[],[],[] ]  
		# }

		totals_by_label = {
			"left": [],
			"keep": [],
			"right": []
		}

		for label in self.classes:
			for i in range(num_vars):
				totals_by_label[label].append([])  # empty array

		# start training
		for x, label in zip(X, Y):
			# process each tuple - as desired
			x = self.process_vars(x)

			# add this data to appt place
			for i, val in enumerate(x):
				totals_by_label[label][i].append(val)

		# Get means and stds for each of the arrays we built.
		# These will be used as priors in the GNB
		means	= []
		stds	= []
		for i in self.classes:
			means.append([])
			stds.append([])
			for arr in totals_by_label[i]:
				mean	= np.mean(arr)
				std		= np.std(arr)
				means[-1].append(mean)
				stds[-1].append(std)

		# store - these are our priors
		self._means = means  ## means[0] = [_, _, _, _] -> for LEFT; means[1] -> for KEEP; means[2] -> for RIGHT
		self._stds  = stds  ## ditto for stds

		## DEbug only
		print('means len:', len(means))
		print('means[0] ', means[0]) 
		for j in means:
			print(j)
		print('len LEFT[0]', len(totals_by_label["left"][0]))
		for lab in ['left', 'keep', 'right']:
			print('Len: ', lab)
			print(len(totals_by_label[lab][0]))
		## end debug

	def _predict(self, obs):
		'''
			private method - used to assign probability of each class for one observation tuple
		'''
		obs = self.process_vars(obs)
		probs = []

		for (means, stds, label) in zip(self._means, self._stds, self.classes):
			product = 1
			for mu, sigma, o in zip(means, stds, obs):
				likelihood = gaussian_probability(o, mu, sigma)
				product   *= likelihood
			probs.append(product)
		
		t = sum(probs)
		return [p/t for p in probs]

	def predict(self, observation):
		"""
		Once trained, this method is called and expected to return 
		a predicted behavior for the given observation.

		INPUTS

		observation - a 4 tuple with s, d, s_dot, d_dot.
		  - Example: [3.5, 0.1, 8.5, -0.2]

		OUTPUT

		A label representing the best guess of the classifier. Can
		be one of "left", "keep" or "right".
		"""
		# TODO - complete this
		# fetch prob for each class
		probs = self._predict(observation)
		idx 	= 0	  # index of best probability
		best_p = 0
		for i, p in enumerate(probs):
			if p > best_p:
				best_p = p
				idx    = i

		names = ['left', 'keep', 'right']
		return names[idx]