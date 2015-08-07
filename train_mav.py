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

class transition_model:
	trans_counts = {}

	def __init__(self):
		self.trans_counts = {}

	#Takes 'classic_xml variety'
	#direct=os.environ['TRANSITION_DIR']
	def load(self, direct):
		import ParseLGL
		conn = psycopg2.connect(os.environ['DB_CONN'])

		cur = conn.cursor()
		m = 0
		for xml_infile in os.listdir(direct):
			
			print xml_infile
			m += 1
			wordref, toporef, domain = ParseLGL.parse_xml(os.path.join(direct, xml_infile))

			transition_dict = featurize_transition(wordref, toporef, domain, cur)
			print transition_dict
			#print toporef
			#print len(wordref)

		conn.close()

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
			if m < 300:
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

	#generate probability of word in region assuming each geog has independent distribution
	def bigram_prob_indep(self, bigram, smoothing="simple-interp", lamb=.5):
		probdict = {}
		if smoothing=='simple-interp':
			firstword = bigram.split('|')[0]
			secondword = bigram.split('|')[1]
			for geocat in self.obs_counts:

				if geocat != 'global':
					#Add in some absolute discounting
					bi_discount = self.obs_counts[geocat]['$BI_TOTAL$'] / self.obs_counts[geocat]['$BI_TYPES$']
					uni_discount = self.obs_counts[geocat]['$UNI_TOTAL$'] / self.obs_counts[geocat]['$UNI_TYPES$']
					print "============="
					print "bi_discount", bi_discount
					print "uni_discount", uni_discount

					uni_prob_first = (max(self.obs_counts[geocat].get(firstword, 0.0) - uni_discount, 0.0) / self.obs_counts[geocat]['$UNI_TOTAL$'])
					uni_prob_second = (max(self.obs_counts[geocat].get(secondword, 0.0) - uni_discount, 0.0) / self.obs_counts[geocat]['$UNI_TOTAL$'])
					#bi_discount = 2.0
					bi_prob = max(self.obs_counts[geocat].get(bigram, 0.0)-bi_discount, 0.0) / self.obs_counts[geocat].get(firstword, 1.0)
					#print geocat, bi_prob, uni_prob_first, uni_prob_second
					#print "bigram-count: ", self.obs_counts[geocat].get(bigram, 0.0), "unigram-first:", self.obs_counts[geocat].get(firstword, 0.0), "unigram-total:", self.obs_counts[geocat]['$UNI_TOTAL$']
					interp_prob = lamb * bi_prob + (((1.0 - lamb)/2.0) * uni_prob_first) + (((1.0 - lamb)/2.0) * uni_prob_second)
					probdict[geocat] = interp_prob
		return probdict

	#generate probability given bigram. assumes all geographies are in same distribution
	def bigram_prob(self, bigram, smoothing="simple-interp", lamb=.5):
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

import io

def get_distbin(Dist_Bins, dist_transition):
	for b in Dist_Bins:
		if dist_transition <= Dist_Bins[b][1] and dist_transition >= Dist_Bins[b][0]:
			return b
	return 'global'

def get_tokenbin(Dist_Bins, dist_transition):
	for b in Dist_Bins:
		if dist_transition <= Dist_Bins[b][1] and dist_transition >= Dist_Bins[b][0]:
			return b
	return 'document'

#Get region given latitutde, longitude, DB cur
def getRegion(lat, lon, cur):
	
	SQL_REGION = "SELECT p2.region_name, ST_Distance(ST_GeographyFromText('SRID=4326;POINT(%s %s)'), p2.geog)/1000.0 from customgrid as p2;" % (lon, lat)
	cur.execute(SQL_REGION)
	results = cur.fetchall()
	results.sort(key=lambda x: x[1])
	#print results
	return results[0][0]


def featurize_transition(wordref, toporef, domain, cur):
	j = 0
	Dist_Bins = {'local':[0.0, 161.0], 'region':[161.1, 500.0], 'country':[500.1, 1500.0], 'global':[1501.1, 15000.0]}
	Token_Bins = {'adjacent':[0, 4], 'sentence':[5, 25], 'paragraph':[26, 150], 'document':[151, 4000]}
	transition_dict = {}
	prev_region = '#START#'
	for i in sorted(toporef.keys()):
		j += 1
		#print i, toporef[i]
		lon = toporef[i][1]['long']
		lat = toporef[i][1]['lat']
		toponym = toporef[i][0]
		docid = toporef[i][-2]
		wid = i
		regions = []

		#print domain, results[0][0]

		if j > 1:
			prev_lon = last_topo[1]['long']
			prev_lat = last_topo[1]['lat']
			prev_wid = last_topo[-1]
			prev_docid = last_topo[-2]
			if last_topo[0].lower() != toporef[i][0].lower():
				print last_topo[0], "->", toporef[i][0]
				#SQL = "SELECT ST_DISTANCE(ST_GeographyFromText('SRID=4326;POINT(%s %s)'), ST_GeographyFromText('SRID=4326;POINT(%s %s)'));" % (lon, lat, prev_lon, prev_lat)
				SQL = "SELECT ST_DISTANCE(p1.polygeog2, p2.polygeog2) from lgl_dev_classic as p1, lgl_dev_classic as p2 where p1.polygeog2 is not null and p2.polygeog is not null and p2.wid = %s and p2.docid = %s and p1.wid = %s and p1.docid = %s;" % ('%s', '%s', '%s', '%s')
				cur.execute(SQL, (prev_wid, prev_docid, i, docid))
				results = cur.fetchall()
				if len(results) > 0:
					print "1st", results
				else:			
					SQL = "SELECT ST_DISTANCE(p1.polygeog2, ST_GeographyFromText('SRID=4326;POINT(%s %s)')) from lgl_dev_classic as p1 where p1.polygeog2 is not null and p1.wid = %s and p1.docid = %s;" % (prev_lon, prev_lat, '%s', '%s')
					cur.execute(SQL, (prev_wid, prev_docid))
					results = cur.fetchall()
					if len(results) > 0:
						print "2nd", results
					else:
						SQL = "SELECT ST_DISTANCE(ST_GeographyFromText('SRID=4326;POINT(%s %s)'), p2.polygeog2) from lgl_dev_classic as p2 where p2.polygeog2 is not null and p2.wid = %s and p2.docid = %s;" % (lon, lat, '%s', '%s')
						cur.execute(SQL, (i, docid))
						results = cur.fetchall()
						if len(results) > 0:
							print "3rd", results
						else: 
							SQL = "SELECT ST_DISTANCE(ST_GeographyFromText('SRID=4326;POINT(%s %s)'), ST_GeographyFromText('SRID=4326;POINT(%s %s)'));" % (lon, lat, prev_lon, prev_lat)
							cur.execute(SQL)
							results = cur.fetchall()


				dist_transition = results[0][0]/1000.0
				token_dist = i - last_topo[-1]
				print "Transition Dist:", dist_transition
				print "Token Dist:", token_dist 
				distbin = get_distbin(Dist_Bins, dist_transition)
				print "Dist Bin:", distbin
				tokebin = get_tokenbin(Token_Bins, token_dist)
				print "Token Bin:", tokebin

				
				current_region = getRegion(lat, lon, cur)

				if prev_region not in transition_dict:
					transition_dict[prev_region] = {}
				transition_dict[prev_region][current_region] = (transition_dict[prev_region].get(current_region, 0) + 1)  

			

		last_topo = toporef[i]
		last_topo[-1] = i
		if j > 1:
			prev_region = current_region
	current_region = "#END#"
	if prev_region not in transition_dict:
		transition_dict[prev_region] = {}
	transition_dict[prev_region][current_region] = (transition_dict[prev_region].get(current_region, 0) + 1)
	return transition_dict 

def test_LGL(LM, directory="/home/grant/devel/TopCluster/LGL/articles/dev_classicxml"):

	import ParseLGL

	out_test = "test_output.txt"

	ot = io.open(out_test, 'w', encoding='utf-8')

	conn = psycopg2.connect(os.environ['DB_CONN'])
	cur = conn.cursor()

	cor = 0
	total = 0
	for f in os.listdir(directory):
		print f
		wordref, toporef, domain = ParseLGL.parse_xml(os.path.join(directory, f))
		topo_context_dict = ParseLGL.getTopoContexts(wordref, toporef, window=1)
		#print topo_context_dict
		for t in topo_context_dict:
			print "===="
			ot.write(u"====\n")
			#print topo_context_dict[t]['entry']
			geo_logprobs = {}
			for c in topo_context_dict[t]['context']:
				if '|' in c:
					plist = LM.bigram_prob_indep(c)
					for region in plist:
						try:
							ot.write(unicode(c) + u' ' + unicode(region) + u':' + unicode(plist[region]))
						except:
							ot.write(c.encode('utf-8') + u' ' + unicode(region) + u':' + unicode(plist[region]))			
						if plist[region] > 0.0:
							geo_logprobs[region] = geo_logprobs.get(region, 0.0) + math.log(plist[region])
			ot.write(u'\n')
			problist = geo_logprobs.items()
			problist.sort(key=lambda x: x[1])
			#print problist
			region_name = problist[-1][0].replace('_', ' ')
			region_prob = problist[-1][-1]
			lat = float(topo_context_dict[t]['entry'][1]['lat'])
			lon = float(topo_context_dict[t]['entry'][1]['long'])
			#print region_name
			SQL_ACC = "SELECT ST_Distance(ST_GeographyFromText('SRID=4326;POINT(%s %s)'), p2.geog)/1000.0 from customgrid as p2 where p2.region_name = %s;" % (lon, lat, '%s')
			#print SQL_ACC
			cur.execute(SQL_ACC, (region_name, ))
			returns = cur.fetchall()
			
			#print returns[0], '|', topo_context_dict[t], '|',  region_name, '|', region_prob
			#print problist
			ot.write(unicode([returns[0], topo_context_dict[t], region_name, region_prob]))
			ot.write(u'\n')
			ot.write(unicode(problist))
			ot.write(u'\n')
			#if returns[0][0] == True:
			#	cor += 1
			total += 1
			#if total % 50 == 0:
			#	print cor, "/", total

	ot.close()

	print cor, "/", total


#LM = lang_model()
#LM.load()
#test_LGL(LM, directory="/work/02608/grantdel/corpora/LGL/articles/dev_classicxml")

TM = transition_model()
TM.load(direct="/home/grant/devel/TopCluster/LGL/articles/dev_classicxml")


'''for bg in ['New|York', 'United|States', 'United|Kingdom', 'Texas|State', 'Austin|#MARK#', 'in|Austin']:
	plist =  LM.bigram_prob(bg).items()
	plist.sort(key=lambda x: x[1])
	print bg
	print plist
	print "========="'''
