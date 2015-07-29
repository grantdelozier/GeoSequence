import os
import sys
import json
from collections import defaultdict
import math
import psycopg2

try:
	os.environ['OBSPATH']
except:
	print "ERROR: OBSPATH environment variable is not set"
	sys.exit()
try:
	os.environ['DB_CONN']
except:
	#Looks like "dbname=topodb user=postgres host='localhost' port='5433' password='grant'"
	print "ERROR: DB_CONN string is not set"
	sys.exit()


class lang_model:

	obs_counts = {}

	def __init__(self):
		self.obs_counts = {}

	#load obscounts dict given obs directory
	def load(self, direct=os.environ['OBSPATH']):
		j = 0
		m = 0
		self.obs_counts['global'] = defaultdict(float)

		for f in os.listdir(direct):
			print "Loading", f
			print m, "/", len(os.listdir(direct))
			m += 1
			if m < 25:
				fp = os.path.join(direct, f)
				geocat = f.split('_uni_bigram')[0]
				self.obs_counts[geocat] = {}
				uni_total = 0.0
				bi_types = 0
				bi_total = 0.0
				uni_types = 0

				with open(fp, 'rb') as r:
					rdict = json.loads(r.read())
					for k in rdict:
						j += 1
						if j % 200000 == 0:
							print j
						self.obs_counts[geocat][k] = rdict[k]
						self.obs_counts['global'][k] += rdict[k]
						if '|' not in k:
							uni_total += rdict[k]
							uni_types += 1
						else:
							bi_types += 1
							bi_total += rdict[k]
							#second_word = k.split('|')[1]
							#first_word = k.split('|')[0]
							#self.obs_counts[geocat]['$SECOND_WORD$|'+second_word] = self.obs_counts[geocat].get('$SECOND_WORD$|'+second_word, 0) + rdict[k]
							#self.obs_counts[geocat]['$FIRST_WORD$|'+first_word] = self.obs_counts[geocat].get('$FIRST_WORD$|'+first_word, 0) + rdict[k]
					self.obs_counts[geocat]['$UNI_TOTAL$'] = uni_total
					self.obs_counts[geocat]['$UNI_TYPES$'] = uni_types
					self.obs_counts[geocat]['$BI_TYPES$'] = bi_types
					self.obs_counts[geocat]['$BI_TOTAL$'] = bi_total



	#generate probability given unigram
	def unigram_prob(smoothing="kneser-ney"):
		pass

	#generate probability given bigram
	def bigram_prob(self, bigram, smoothing="simple-interp", lamb=.6):
		probdict = {}
		if smoothing=='kneser-ney':
			firstword = bigram.split('|')[0]
			secondword = bigram.split('|')[1]
			'''for geocat in self.obs_counts:
				discount = self.obs_counts[geocat]['$BI_TOTAL$'] / self.obs_counts[geocat]['$BI_TYPES$']
				normalizing_constant = (discount / self.obs_counts[geocat].get(firstword, 1.0)) * float(self.obs_counts[geocat]['$FIRST_WORD$|'+firstword])
				right_term = normalizing_constant * (float(self.obs_counts[geocat].get('$SECOND_WORD$|'+secondword, 0.0)) / float(self.obs_counts[geocat]['$BI_TYPES$']))
				left_term = max(self.obs_counts[geocat].get(bigram, 0.0)-discount, 0.0) / self.obs_counts[geocat].get(firstword, 1.0)
				print geocat, left_term, right_term, normalizing_constant, discount
				probdict[geocat] = right_term + left_term'''
		elif smoothing=='simple-interp':
			firstword = bigram.split('|')[0]
			secondword = bigram.split('|')[1]
			for geocat in self.obs_counts:
				if geocat != 'global':
					uni_prob_first = self.obs_counts[geocat].get(firstword, 0.0) / self.obs_counts['global'].get(firstword, 1.0) 
					uni_prob_second = self.obs_counts[geocat].get(secondword, 0.0) / self.obs_counts['global'].get(secondword, 1.0)
					bi_prob =  self.obs_counts[geocat].get(bigram, 0.0) / self.obs_counts['global'].get(bigram, 1.0)
					interp_prob = lamb * bi_prob + (((1.0 - lamb)/2.0) * uni_prob_first) + (((1.0 - lamb)/2.0) * uni_prob_second)
					probdict[geocat] = interp_prob
			'''for geocat in self.obs_counts:

				#Add in some absolute discounting
				bi_discount = self.obs_counts[geocat]['$BI_TOTAL$'] / self.obs_counts[geocat]['$BI_TYPES$']
				uni_discount = self.obs_counts[geocat]['$UNI_TOTAL$'] / self.obs_counts[geocat]['$UNI_TYPES$']
				print "============="
				print "bi_discount", bi_discount
				print "uni_discount", uni_discount

				uni_prob_first = (max(self.obs_counts[geocat].get(firstword, 0.0) - uni_discount, 0.0) / self.obs_counts[geocat]['$UNI_TOTAL$'])
				uni_prob_second = (max(self.obs_counts[geocat].get(secondword, 0.0) - uni_discount, 0.0) / self.obs_counts[geocat]['$UNI_TOTAL$'])
				#bi_discount = 2.0
				bi_prob = max(self.obs_counts[geocat].get(bigram, 0.0)-bi_discount, 0.0) / self.obs_counts[geocat].get(firstword, 0.0)
				print geocat, bi_prob, uni_prob_first, uni_prob_second
				print "bigram-count: ", self.obs_counts[geocat].get(bigram, 0.0), "unigram-first:", self.obs_counts[geocat].get(firstword, 0.0), "unigram-total:", self.obs_counts[geocat]['$UNI_TOTAL$']
				interp_prob = lamb * bi_prob + (((1.0 - lamb)/2.0) * uni_prob_first) + (((1.0 - lamb)/2.0) * uni_prob_second)
				probdict[geocat] = interp_prob'''
		return probdict


def test_LGL(LM, directory="/home/grant/devel/TopCluster/LGL/articles/dev_classicxml")

	import ParseLGL

	conn = psycopg2.connect(os.environ['DB_CONN'])
	cur = conn.cursor()

	cor = 0
	total = 0
	for f in os.listdir(directory):
		print f
		wordref, toporef, domain = ParseLGL.parse_xml(os.path.join(directory, f))
		topo_context_dict = ParseLGL.getTopoContexts(wordref, toporef, window=4)
		#print topo_context_dict
		for t in topo_context_dict:
			#print topo_context_dict[t]['entry']
			geo_logprobs = {}
			for c in topo_context_dict[t]['context']:
				if '|' in c:
					plist = LM.bigram_prob(c)
					for region in plist:
						if plist[region] > 0.0:
							geo_logprobs[region] = geo_logprobs.get(region, 0.0) + math.log(plist[region])
			problist = geo_logprobs.items()
			problist.sort(key=lambda x: x[1])
			#print problist
			region_name = problist[-1][0].replace('_', ' ')
			region_prob = problist[-1][-1]
			lat = float(topo_context_dict[t]['entry'][1]['lat'])
			lon = float(topo_context_dict[t]['entry'][1]['long'])
			#print region_name
			SQL_ACC = "SELECT ST_DWITHIN(ST_GeographyFromText('SRID=4326;POINT(%s %s)'), p2.geog, 1000) from customgrid as p2 where p2.region_name = %s;" % (lon, lat, '%s')
			#print SQL_ACC
			cur.execute(SQL_ACC, (region_name, ))
			returns = cur.fetchall()
			if returns[0][0] == True:
				cor += 1
			total += 1
			if total % 50 == 0:
				print cor, "/", total

	print cor, "/", total


LM = lang_model()
LM.load()
test_LGL(LM)

'''for bg in ['New|York', 'United|States', 'United|Kingdom', 'Texas|State', 'Austin|#MARK#', 'in|Austin']:
	plist =  LM.bigram_prob(bg).items()
	plist.sort(key=lambda x: x[1])
	print bg
	print plist
	print "========="'''