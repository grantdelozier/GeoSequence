import os
import sys
import json
from collections import defaultdict
import math
import io
import scipy
from scipy.sparse import csr_matrix


try:
	import psycopg2
except:
	sys.path.insert(0, '/home/02608/grantdel/pythonlib/lib/python2.7/site-packages')
	import psycopg2
	from sklearn import linear_model

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

class transition_model_discrim:
	trans_data = []
	feature_index = {}
	label_index = {}
	custom_regions = []
	country_names = []

	def __init__(self):
		self.trans_counts = {}

	def load(self, direct):
		import ParseLGL
		conn = psycopg2.connect(os.environ['DB_CONN'])

		self.load_custom_regions()
		self.load_country_names()

		cur = conn.cursor()
		m = 0
		for xml_infile in os.listdir(direct):
			
			print xml_infile
			m += 1
			wordref, toporef, domain = ParseLGL.parse_xml(os.path.join(direct, xml_infile))
			self.trans_data = featurize_transition_discrim(wordref, toporef, domain, cur, self.trans_data, self.country_names)

		conn.close()

	def train(self):
		X = []
		Y = []

		feature_index, label_index = {}, {}
		i, j, y = 0, 0, 0
		row_indexes, col_indexes, values, labels = [], [], [], []
		feature_index[""] = 0
		i += 1
		for line in self.trans_data:
			if len(line) > 0:
				label = line[0]
				feats = line[1]
			row_indexes.append(len(values))
			# Add the intercept constant
			col_indexes.append(0)
			values.append(1.0)
			j += 1
			if label not in label_index:
				label_index[label] = y
				y += 1
			labels.append(label_index[label])
			for f in feats:
				if f not in feature_index:
					feature_index[f] = i
					i += 1
				col_indexes.append(feature_index[f])
				values.append(1.0)
		row_indexes.append(row_indexes[-1] + 1)

		X = csr_matrix((scipy.array(values, dtype=scipy.float64), scipy.array(col_indexes),
						scipy.array(row_indexes)), shape=(j, i), dtype=scipy.float64)
		Y = scipy.array(labels)

		model = linear_model.LogisticRegression(penalty='l2', fit_intercept=False, C=10.0, solver='newton-cg')
		model.fit(X, Y)
		coefs = model.coef_

		print label_index
		print feature_index
		print coefs.tolist()

		self.label_index = label_index
		self.feature_index = feature_index
		self.weights = coefs.tolist()

	def log_prob_dict(self, prev_toponame, current_toponame, toke_dist):
		Token_Bins = {'adjacent':[0, 4], 'sentence':[5, 25], 'paragraph':[26, 150], 'document':[151, 4000]}

		feature_vector = np.zeros(len(self.feature_index))
		obs_features = []
		if prev_toponame.lower() == current_toponame.lower():
			obs_features.append('SAME')
		else:
			obs_features.append('NOT-SAME')

		tokebin = get_tokenbin(Token_Bins, token_dist)
		if tokebin in self.feature_index:
			obs_features.append(tokebin)

		prob_dict = {}
		for label in self.label_index:
			label_sum = 0.0
			label_sum += self.weights[self.label_index[label]][0]
			for feat in obs_features:
				label_sum += self.weights[self.label_index[label]][self.feature_index[feat]]
			prob_dict[label] = math.exp(label_sum) / (1.0 + math.exp(label_sum))
		return log_prob_dict


	def load_country_names(self):
		conn = psycopg2.connect(os.environ['DB_CONN'])
		cur = conn.cursor()
		country_names = []
		SQL = "SELECT p1.name, p1.postal, p1.abbrev, p1.name_long, p1.altnames from countries_2012 as p1;"
		cur.execute(SQL)
		results = cur.fetchall()

		for row in results:
			for name in row:
				if name != None:
					if ',' in name:
						for nm in name.split(','):
							country_names.append(unicode(nm.decode('utf-8').lower()))
					else:
						country_names.append(unicode(name.decode('utf-8').lower()))

		#for n in country_names:
		#	print n

		self.country_names = set(country_names)

		conn.close()

	def load_custom_regions(self):
		conn = psycopg2.connect(os.environ['DB_CONN'])
		cur = conn.cursor()

		SQL = "SELECT p1.region_name from customgrid as p1;"
		cur.execute(SQL)
		returns = cur.fetchall()
		for name in returns:
			self.custom_regions.append(name[0])
		conn.close()

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

			self.trans_counts = featurize_transition_gen(wordref, toporef, domain, cur)
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
		conn.close()

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


def viterbi_discrim(obs, states, TM, LM):
	pass

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

def getRegionBin(current_region, prev_region, cur):
	#print current_region
	#print prev_region
	if current_region == prev_region:
		return 'SAME'
	SQL_DIST  = "SELECT ST_DWithin(p1.geog, p2.geog, 161000.0) from customgrid as p1, customgrid as p2 where p1.region_name = %s and p2.region_name = %s;"
	cur.execute(SQL_DIST, (current_region, prev_region))
	results = cur.fetchall()
	#print results
	if results[0][0] == True:
		return "LOCAL/ADJACENT"
	SQL_DIST  = "SELECT ST_DWithin(p1.geog, p2.geog, 1500000.0) from customgrid as p1, customgrid as p2 where p1.region_name = %s and p2.region_name = %s;"
	cur.execute(SQL_DIST, (current_region, prev_region))
	results = cur.fetchall()
	if results[0][0] == True:
		return "COUNTRY"
	return "CONTINENT/GLOBAL"

#RETURN COUNTRY if toponym is a name or alt name of a country
def isCountryName(toponym, country_names):
	if toponym.lower() in country_names:
		return 'COUNTRY'
	else:
		return 'NOT-COUNTRY'


#Get region given latitutde, longitude, DB cur
def getRegion(lat, lon, cur):
	
	SQL_REGION = "SELECT p2.region_name, ST_Distance(ST_GeographyFromText('SRID=4326;POINT(%s %s)'), p2.geog)/1000.0 from customgrid as p2;" % (lon, lat)
	cur.execute(SQL_REGION)
	results = cur.fetchall()
	results.sort(key=lambda x: x[1])
	#print results
	return results[0][0]


def featurize_transition_discrim(wordref, toporef, domain, cur, country_names):
	j = 0
	#Dist_Bins = {'local':[0.0, 161.0], 'region':[161.1, 500.0], 'country':[500.1, 1500.0], 'global':[1501.1, 15000.0]}
	Token_Bins = {'adjacent':[0, 4], 'sentence':[5, 25], 'paragraph':[26, 150], 'document':[151, 4000]}
	prev_region = '#START#'

	transition_data = []

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

		current_region = getRegion(lat, lon, cur)

		if j > 1:
			prev_lon = last_topo[1]['long']
			prev_lat = last_topo[1]['lat']
			prev_wid = last_topo[-1]
			prev_docid = last_topo[-2]
			prev_toponame = last_topo[0]
			
			print last_topo[0], "->", toporef[i][0]
			
			token_dist = i - last_topo[-1]
			tokebin = get_tokenbin(Token_Bins, token_dist)

			label = getRegionBin(current_region, prev_region, cur)

			country_name_feat = isCountryName(toponym, country_names)

			if prev_toponame.lower() == toponym.lower():
				sameTopo = 'SAME'
			else:
				sameTopo = 'NOT-SAME'


			transition_data.append([label, [tokebin, sameTopo, country_name_feat]])
			print transition_data[-1]

			
		last_topo = toporef[i]
		last_topo[-1] = i
		if j > 0:
			prev_region = current_region

	return transition_data

def featurize_transition_gen(wordref, toporef, domain, cur, transition_dict):
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

def test_viterbi_poly(LM, TM, directory="/work/02608/grantdel/corpora/LGL/articles/dev_testsplit1", poly_table_name = "lgl_dev_classic"):
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

				did = pred[1][-2]
				wid = pred[1][-1]
				#print "did: ", did
				#print "wid: ", wid

				SQL_ACC = "SELECT ST_DWithin(p1.polygeog2, p2.geog, 160000) from customgrid as p2, %s as p1 where p2.region_name = %s and p1.docid = %s and p1.wid = %s;" % (poly_table_name, '%s', '%s', '%s')				#print SQL_ACC
				cur.execute(SQL_ACC, (pred_region, did, wid))
				returns = cur.fetchall()
				#print returns
				if returns[0][0] == None:
					SQL_POINT = "SELECT ST_Distance(ST_GeographyFromText('SRID=4326;POINT(%s %s)'), p2.geog)/1000.0 from customgrid as p2 where p2.region_name = %s;" % (lon, lat, '%s')
					#print SQL_ACC
					cur.execute(SQL_POINT, (pred_region, ))
					returns = cur.fetchall()
					if returns[0][0] < 160.0:
						cor += 1
						#print "backed off to point acc and found CORRECT"
				elif returns[0][0] == True:
					cor += 1
				total += 1
				#print "viterbi poly total: ", total

				try:
					ot.write(unicode(pred_region) + u'|' +  unicode(pred[1][0]) + u'|' + unicode(lat) + u'|' + unicode(lon) + u'|' + unicode(returns[0][0]))
					ot.write(u'\n')
				except:
					print "=========="
					print "error writing"
					print pred

	print "VITERBI ACC POLY:"
	print cor, "/", total
	print float(cor)/float(total)

	ot.close()
	conn.close()

def test_viterbi_discrim(LM, TM, directory="/work/02608/grantdel/corpora/LGL/articles/dev_testsplit1"):
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
		obs = [[topo, topo_context_dict[topo]['context'].keys()] for topo in ordered_tkeys]
		print obs
		#print "==="
		#print "obs"
		#print obs
		#print "==="
		states = TM.custom_regions
		if len(obs) > 0:
			viterbi_discrim(obs, states, TM, LM)

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

def test_pureLM_poly(LM, directory="/home/grant/devel/TopCluster/LGL/articles/dev_testsplit1", poly_table_name="lgl_dev_classic"):
	import ParseLGL

	out_test = "test_output_poly.txt"

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

			did = toporef[t][-2]
			wid = toporef[t][-1]
			#print "did: ", did
			#print "wid: ", wid
			#print "===="
			ot.write(u"====\n")
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

			region_name = problist[-1][0]
			region_prob = problist[-1][-1]
			lat = float(topo_context_dict[t]['entry'][1]['lat'])
			lon = float(topo_context_dict[t]['entry'][1]['long'])
			#print region_name
			#SQL_ACC = "SELECT ST_Distance(p1.polygeog2, p2.geog)/1000.0 from customgrid as p2, %s as p1 where p2.region_name = %s;" % (poly_table_name, '%s')
			SQL_ACC = "SELECT ST_DWithin(p1.polygeog2, p2.geog, 160000) from customgrid as p2, %s as p1 where p2.region_name = %s and p1.docid = %s and p1.wid = %s;" % (poly_table_name, '%s', '%s', '%s')
			#print SQL_ACC
			cur.execute(SQL_ACC, (region_name, did, wid))
			returns = cur.fetchall()
			
			if returns[0][0] == None:
				SQL_POINT = "SELECT ST_Distance(ST_GeographyFromText('SRID=4326;POINT(%s %s)'), p2.geog)/1000.0 from customgrid as p2 where p2.region_name = %s;" % (lon, lat, '%s')
				#print SQL_ACC
				cur.execute(SQL_POINT, (region_name, ))
				returns = cur.fetchall()
				if returns[0][0] < 160.0:
					cor += 1
					#print "backed off to point acc and found CORRECT"
			elif returns[0][0] == True:
				cor += 1
			total += 1

			#print returns[0], '|', topo_context_dict[t], '|',  region_name, '|', region_prob
			#print problist
			ot.write(unicode([returns[0], topo_context_dict[t], region_name, region_prob]))
			ot.write(u'\n')
			ot.write(unicode(problist))
			ot.write(u'\n')
			#print returns

			#print "pure LM poly total: ", total

	ot.close()
	conn.close()

	print "PURE LM ACC POLY:"
	print cor, "/", total
	print float(cor)/float(total)

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
			#print "viterbi poly total: ", total

	ot.close()
	conn.close()

	print "PURE LM ACC:"
	print cor, "/", total
	print float(cor)/float(total)


LM = lang_model()
LM.load()

TM = transition_model_discrim()
TM.load("/work/02608/grantdel/corpora/LGL/articles/dev_trainsplit4")
TM.train()
#test_pureLM(LM, directory="/work/02608/grantdel/corpora/trconllf/dev_testsplit5")
test_viterbi_discrim(LM, TM, directory="/work/02608/grantdel/corpora/trconllf/dev_testsplit4")

#test_pureLM_poly(LM, directory="/work/02608/grantdel/corpora/LGL/articles/dev_testsplit4", poly_table_name="lgl_dev_classic")
#test_viterbi_poly(LM, TM, directory="/work/02608/grantdel/corpora/LGL/articles/dev_testsplit4", poly_table_name="lgl_dev_classic")



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
