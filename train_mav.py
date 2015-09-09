import os
import sys
import json
from collections import defaultdict
import math
import psycopg2
import io


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
	custom_regions = []

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

			self.trans_counts = featurize_transition(wordref, toporef, domain, cur, self.trans_counts)
			#print self.trans_counts
			#print toporef
			#print len(wordref)

		#print self.trans_counts
		conn.close()

		self.load_custom_regions()

	def load_custom_regions(self):
		conn = psycopg2.connect(os.environ['DB_CONN'])
		cur = conn.cursor()

		SQL = "SELECT p1.region_name from customgrid as p1;"
		cur.execute(SQL)
		returns = cur.fetchall()
		for name in returns:
			self.custom_regions.append(name[0])

	def binomial_prob_dict(self):
		p_dict = {}
		for k in sorted(self.trans_counts.keys()):
			p_dict[k] = {}
			for r in self.custom_regions:
				p_dict[k][r] = self.binomial_prob(k, r)
		return p_dict

	#Additive smoothing
	def binomial_prob(self, prev_region, current_region, additive_smoothing=3.0):
		if prev_region not in self.trans_counts:
			cur_region_n = additive_smoothing
		else:
			cur_region_n = self.trans_counts[prev_region].get(current_region, 0.0) + additive_smoothing
		prev_total = 0.0
		for r in self.custom_regions:
			if prev_region in self.trans_counts:
				prev_total += (self.trans_counts[prev_region].get(r, 0.0) + additive_smoothing)
			else:
				prev_total += additive_smoothing
		prob = cur_region_n / float(prev_total)
		return prob

class lang_model:

	obs_counts = {}
	custom_regions = []

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
				geocat = f.split('_uni_bigram')[0].replace('_', ' ')
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
							if rdict[k] < 0.0:
								print "Shit is fucked up"
								print geocat, k, rdict[k]
								sys.exit()
							#second_word = k.split('|')[1]
							#first_word = k.split('|')[0]
							#self.obs_counts[geocat]['$SECOND_WORD$|'+second_word] = self.obs_counts[geocat].get('$SECOND_WORD$|'+second_word, 0) + rdict[k]
							#self.obs_counts[geocat]['$FIRST_WORD$|'+first_word] = self.obs_counts[geocat].get('$FIRST_WORD$|'+first_word, 0) + rdict[k]
					self.obs_counts[geocat]['$UNI_TOTAL$'] = uni_total
					self.obs_counts[geocat]['$UNI_TYPES$'] = uni_types
					self.obs_counts[geocat]['$BI_TYPES$'] = bi_types
					self.obs_counts[geocat]['$BI_TOTAL$'] = bi_total
		self.load_custom_regions()

	def load_custom_regions(self):
		conn = psycopg2.connect(os.environ['DB_CONN'])
		cur = conn.cursor()

		SQL = "SELECT p1.region_name from customgrid as p1;"
		cur.execute(SQL)
		returns = cur.fetchall()
		for name in returns:
			self.custom_regions.append(name[0])


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
	def bigram_prob(self, bigram, smoothing="simple-interp-laplace", lamb=.5):
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
		elif smoothing=="simple-interp-laplace":
			firstword = bigram.split('|')[0]
			secondword = bigram.split('|')[1]
			for geocat in self.custom_regions:
				if geocat != 'global':
					uni_prob_first = (self.obs_counts[geocat].get(firstword, 0.0) + 1.0) / float(self.obs_counts['global'].get(firstword, 0.0) + (len(self.custom_regions)-1.0))
					uni_prob_second = (self.obs_counts[geocat].get(secondword, 0.0) + 1.0) / float(self.obs_counts['global'].get(secondword, 0.0) + (len(self.custom_regions)-1.0))
					bi_prob =  (self.obs_counts[geocat].get(bigram, 0.0) + 1.0) / float(self.obs_counts['global'].get(bigram, 0.0) + (len(self.custom_regions)-1.0))
					interp_prob = (lamb * bi_prob) + (((1.0 - lamb)/2.0) * uni_prob_first) + (((1.0 - lamb)/2.0) * uni_prob_second)
					probdict[geocat] = interp_prob
					if interp_prob < 0.0 or interp_prob > 1.0:
						print "Shit is fucked up"
						print bigram
						print "bigram: ", bi_prob
						print "interp: ", interp_prob
						numerator = (self.obs_counts[geocat].get(bigram, 0.0) + 1.0)
						denom = float(self.obs_counts['global'].get(bigram, 0.0) + (len(self.custom_regions)-1.0))
						denom_part1 = self.obs_counts['global'].get(bigram, 0.0)
						denom_part2 = (len(self.custom_regions)-1.0)
						print numerator
						print denom
						print denom_part1
						print denom_part2
						sys.exit()								
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


def get_emission_dict(LM, context_list):
	emission_dict = {}
	for c in context_list:
		if '|' in c:
			plist = LM.bigram_prob(c)
			for region in plist:
				if plist[region] < 0.0 or plist[region] > 1.0:
					print "This shit is broken as fuck"
					print region, plist[region]
					sys.exit()
				else:
					emission_dict[region] = emission_dict.get(region, 0.0) + math.log(plist[region])

	return emission_dict


#states = TM.custom_regions
#obs = list of observations
#TM = Transition Model
#LM = Language Model
def viterbi(obs, states, TM, LM):
    V = [{}]
    path = {}
    
    # Initialize base cases (t == 0)
    emission_dict = get_emission_dict(LM, obs[0])
    for y in states:
        V[0][y] = math.log(TM.binomial_prob('#START#', y)) + emission_dict[y]
        path[y] = [y]
    
    # Run Viterbi for t > 0
    for t in range(1, len(obs)):
        V.append({})
        newpath = {}
        emission_dict = get_emission_dict(LM, obs[t])

        for y in states:
            #print emission_dict[y]
            #print math.log(emission_dict[y])
            #for j in states:
            #    print TM.binomial_prob(j, y)
            #    print math.log(TM.binomial_prob(j, y))
            (prob, state) = max((V[t-1][y0] + math.log(TM.binomial_prob(y0, y)) + emission_dict[y], y0) for y0 in states)
            V[t][y] = prob
            newpath[y] = path[state] + [y]

        # Don't need to remember the old paths
        path = newpath
    n = 0           # if only one element is observed max is sought in the initialization values
    if len(obs) != 1:
        n = t
    #print_dptable(V)
    (prob, state) = max((V[n][y], y) for y in states)
    return (prob, path[state])

def print_dptable(V):
    s = "    " + " ".join(("%7d" % i) for i in range(len(V))) + "\n"
    for y in V[0]:
        s += "%.5s: " % y
        s += " ".join("%.7s" % ("%f" % v[y]) for v in V)
        s += "\n"
    print(s)

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


def featurize_transition(wordref, toporef, domain, cur, transition_dict):
	j = 0
	Dist_Bins = {'local':[0.0, 161.0], 'region':[161.1, 500.0], 'country':[500.1, 1500.0], 'global':[1501.1, 15000.0]}
	Token_Bins = {'adjacent':[0, 4], 'sentence':[5, 25], 'paragraph':[26, 150], 'document':[151, 4000]}
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
			
			#print last_topo[0], "->", toporef[i][0]
			#SQL = "SELECT ST_DISTANCE(ST_GeographyFromText('SRID=4326;POINT(%s %s)'), ST_GeographyFromText('SRID=4326;POINT(%s %s)'));" % (lon, lat, prev_lon, prev_lat)
			'''SQL = "SELECT ST_DISTANCE(p1.polygeog2, p2.polygeog2) from lgl_dev_classic as p1, lgl_dev_classic as p2 where p1.polygeog2 is not null and p2.polygeog is not null and p2.wid = %s and p2.docid = %s and p1.wid = %s and p1.docid = %s;" % ('%s', '%s', '%s', '%s')
			cur.execute(SQL, (prev_wid, prev_docid, i, docid))
			results = cur.fetchall()
			if len(results) > 0:
				#print "1st", results
				pass
			else:			
				SQL = "SELECT ST_DISTANCE(p1.polygeog2, ST_GeographyFromText('SRID=4326;POINT(%s %s)')) from lgl_dev_classic as p1 where p1.polygeog2 is not null and p1.wid = %s and p1.docid = %s;" % (prev_lon, prev_lat, '%s', '%s')
				cur.execute(SQL, (prev_wid, prev_docid))
				results = cur.fetchall()
				if len(results) > 0:
					#print "2nd", results
					pass
				else:
					SQL = "SELECT ST_DISTANCE(ST_GeographyFromText('SRID=4326;POINT(%s %s)'), p2.polygeog2) from lgl_dev_classic as p2 where p2.polygeog2 is not null and p2.wid = %s and p2.docid = %s;" % (lon, lat, '%s', '%s')
					cur.execute(SQL, (i, docid))
					results = cur.fetchall()
					if len(results) > 0:
						#print "3rd", results
						pass
					else: 
						SQL = "SELECT ST_DISTANCE(ST_GeographyFromText('SRID=4326;POINT(%s %s)'), ST_GeographyFromText('SRID=4326;POINT(%s %s)'));" % (lon, lat, prev_lon, prev_lat)
						cur.execute(SQL)
						results = cur.fetchall()


			dist_transition = results[0][0]/1000.0
			token_dist = i - last_topo[-1]
			#print "Transition Dist:", dist_transition
			#print "Token Dist:", token_dist 
			distbin = get_distbin(Dist_Bins, dist_transition)
			#print "Dist Bin:", distbin
			tokebin = get_tokenbin(Token_Bins, token_dist)
			#print "Token Bin:", tokebin
			'''

			
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

#topo_context_dict[t] = {'entry':toporef[t], 'context':d}

def test_viterbi(LM, TM, directory="/work/02608/grantdel/corpora/LGL/articles/dev_testsplit1"):

	import ParseLGL

	out_test = "test_output3.txt"

	ot = io.open(out_test, 'w', encoding='utf-8')

	conn = psycopg2.connect(os.environ['DB_CONN'])
	cur = conn.cursor()

	cor = 0
	total = 0
	obs_sequence = []
	for f in os.listdir(directory):
		#print f
		wordref, toporef, domain = ParseLGL.parse_xml(os.path.join(directory, f))
		topo_context_dict = ParseLGL.getTopoContexts(wordref, toporef, window=1)
		ordered_tkeys = sorted(topo_context_dict.keys())
		obs = [topo_context_dict[topo]['context'].keys() for topo in ordered_tkeys]
		#print "==="
		#print "obs"
		#print obs
		#print "==="
		states = TM.custom_regions
		if len(obs) > 0:
			prob, prob_path = viterbi(obs, states, TM, LM)
			zipped_preds = zip(prob_path, [toporef[topo] for topo in ordered_tkeys])
			#print "prob path", zipped_preds

			for pred in zipped_preds:
				pred_region = pred[0]
				lat = float(pred[1][1]['lat'])
				lon = float(pred[1][1]['long'])


				SQL_ACC = "SELECT ST_Distance(ST_GeographyFromText('SRID=4326;POINT(%s %s)'), p2.geog)/1000.0 from customgrid as p2 where p2.region_name = %s;" % (lon, lat, '%s')
				#print SQL_ACC
				cur.execute(SQL_ACC, (pred_region, ))
				returns = cur.fetchall()
				if returns[0][0] < 161.0:
					cor += 1
				total += 1

				try:
					ot.write(unicode(pred_region) + u'|' +  unicode(pred[1][0]) + u'|' + unicode(lat) + u'|' + unicode(lon) + u'|' + unicode(returns[0][0]))
					ot.write(u'\n')
				except:
					print "=========="
					print "error writing"
					print pred

	print "VITERBI ACC:"
	print cor, "/", total
	print float(cor)/float(total)

	ot.close()
	conn.close()


def test_pureLM(LM, directory="/home/grant/devel/TopCluster/LGL/articles/dev_testsplit1"):

	import ParseLGL

	out_test = "test_output.txt"

	ot = io.open(out_test, 'w', encoding='utf-8')

	conn = psycopg2.connect(os.environ['DB_CONN'])
	cur = conn.cursor()

	cor = 0
	total = 0
	for f in os.listdir(directory):
		#print f
		wordref, toporef, domain = ParseLGL.parse_xml(os.path.join(directory, f))
		topo_context_dict = ParseLGL.getTopoContexts(wordref, toporef, window=1)
		#print topo_context_dict
		for t in topo_context_dict:
			#print "===="
			ot.write(u"====\n")
			#print topo_context_dict[t]['entry']
			geo_logprobs = {}
			for c in topo_context_dict[t]['context']:
				if '|' in c:
					plist = LM.bigram_prob(c)
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
			region_name = problist[-1][0]
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
			if returns[0][0] < 161.0:
				cor += 1
			total += 1
			#if total % 50 == 0:
			#	print cor, "/", total

	ot.close()
	conn.close()

	print "PURE LM ACC:"
	print cor, "/", total
	print float(cor)/float(total)


LM = lang_model()
LM.load()

TM = transition_model()
TM.load("/work/02608/grantdel/corpora/trconllf/dev_trainsplit5")
test_pureLM(LM, directory="/work/02608/grantdel/corpora/trconllf/dev_testsplit5")
test_viterbi(LM, TM, directory="/work/02608/grantdel/corpora/trconllf/dev_testsplit5")

'''TM = transition_model()
TM.load(direct="/home/grant/devel/TopCluster/LGL/articles/dev_classicxml")
prob = TM.binomial_prob('southwest united states', 'southeast united states')
print prob
prob = TM.binomial_prob('southeast united states', 'southwest united states')
prob = TM.binomial_prob('southwest united states', 'southwest united states')
print prob
print prob
prob_dict = TM.binomial_prob_dict()
print prob_dict
for start_region in prob_dict:
	print "region sum:", sum([y for x,y in prob_dict[start_region].items()])'''

'''
for bg in ['New|York', 'United|States', 'United|Kingdom', 'Texas|State', 'Austin|#MARK#', 'in|Austin', 'Iraqi|Officials', 'in|Baghdad']:
	plist =  LM.bigram_prob(bg).items()
	plist.sort(key=lambda x: x[1])
	print bg
	print plist
	print "========="
'''
