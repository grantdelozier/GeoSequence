import os
import sys

try:
	os.environ['OBSPATH']
except:
	print "ERROR: OBSPATH environment variable is not set"
	sys.exit()

class nb_model:

	obs_counts = {}

	__init__(self):
		self.obs_counts = {}

	#load obscounts dict given obs directory
	def load(direct=os.environ['OBSPATH']):
		for f in os.listdir(direct):

	#generate probability given unigram
	def unigram_prob():
		pass

	#generate probability given bigram
	def bigram_prob():
		pass

