import os
import sys
import json
from collections import defaultdict
import math
import io
import scipy
from scipy.sparse import csr_matrix
import numpy as np


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

demonyms = set([u'equatorial guineans', u'french southern territories', u'chinese', u'swedish',
 u'malaysians', u'anguillans', u's\xe3o tom\xe9an', u'dutchwomen', u'turkish', u'incas', u'chadian',
  u'iraqi', u'martiniquaises', u'futunans', u'georgians', u'tanzanian', u'burkinab\xe9',
   u'burkinab\xe8', u'azerbaijanis', u'slovenians', u'bermudian', u'bahamians', u'omani',
    u'nigerians', u'saudi', u'azeris', u'mozambicans', u'croatian', u'uruguayan', u'tunisian',
     u'japanese', u'czechs', u'central african', u'letts', u'somalis', u'south africans', u'indian',
      u'poles', u'faroese', u'aruban', u'omanis', u'sahraouis', u'belongers', u'laotians',
       u'netherlanders', u'r\xe9unionnais', u'french', u'barbudan', u'republic of',
        u'\xe5land islanders', u'malawians', u'canaleros', u'canadian', u'vietnamese', u'vanuatuan',
         u'haitians', u'comorans', u'abkhazian', u'chapines', u'panamanian', u'ethiopian',
          u'yemenis', u'south ossetian', u'italians', u'congolese', u'nepalese', u'hongkongers', u'gibraltarians',
           u'syrians', u'belizeans', u'bissau-guineans', u'palestinian', u'spaniards', u'indians', u'samoans',
            u'miquelonnais', u'australian', u'dominicans', u'belgian', u'northern marianans', u'bahamian', u'bulgarian',
             u'brazilians', u'luxembourgers', u'georgian', u'costa ricans', u'luxembourgish', u'israelis', u'albanians',
              u'none', u'icelanders', u'filipino', u'bouvet island', u'democratic republic of the', u'eritreans', u'costa rican',
               u'mauritanians', u'sahrawis', u'bolivians', u'guadeloupe', u'hong kong', u'papua new guineans', u'tajiks', u'nigerien',
                u'burundian', u'botswanan', u'wallis and futuna islanders', u'guamanian', u'kuwaiti', u'montenegrin', u'macanese',
                 u'egyptian', u'slovenian', u'mongolians', u'mongols', u'magyars', u'kyrgyz', u'nauruan', u'south sudanese', u'lao',
                  u'lithuanian', u'fijian', u'maltese', u'kenyans', u'hellenes', u'equatorial guinean', u'guaran\xedes', u'malians',
                   u'moroccans', u'turks and caicos island', u'tokelauans', u'greek', u'burmese', u'kirgiz', u'mexican', u'futunan',
                    u"democratic people's republic of", u'maldivians', u'guanacos', u'palauans', u'austrian', u'argentines', u'gabonese',
                     u'emirian', u'kittitian', u'argentine', u'guatemalan', u'koreans', u'monacans', u'antarctic residents', u'beninois',
                      u'ugandans', u'chileans', u'saint-martinoise', u'iranians', u'maldivian', u'northern irish', u'mauritanian', u'algerian',
                       u'new zealanders', u'pakistanis', u'irishwomen', u'surinamese', u'european', u'namibian', u'guatemalans', u'trinis',
                        u'botswanans', u'slovaks', u'nigerian', u'christmas island', u'honduran', u'andorrans', u'kirghiz', u'caymanian', u'tanzanians',
 u'afghans', u'greeks', u'uzbekistani', u'mcdonald islands', u'sahraouian', u'barbudans', u'dutch', u'britons', u'kiwis',
 u'malagasy', u'french guianese', u'mon\xe9gasque', u'abkhaz', u'timorese', u'egyptians', u'bhutanese', u'northern marianan',
 u'qataris', u'vincentians', u'french polynesian', u'norwegian', u'british', u's\xe3o tom\xe9ans', u'somalian', u'barbadian',
 u'american', u'belgians', u'sierra leonean', u"people's republic of", u'cubans', u'christmas islanders', u'arubans', u'ayisyen',
u'micronesian', u'svalbard', u'gambians', u'americans', u'spanish', u'antarctic', u'tongan', u'cambodians', u'r\xe9unionnaises',
   u'palauan', u'cabo verdean', u'basotho (singular mosotho)', u'mauritians', u'federated states of', u'mon\xe9gasques', u'sint eustatius',
    u'gibraltar', u'danish', u'solomon islanders', u'herzegovinian', u'burundians', u'greenlandic', u'sri lankans', u'saudi arabians', u'eritrean',
     u'saint helenian', u'thai', u'moldovan', u'ghanaian', u'pakistani', u'barundi', u'colombians', u'iranian', u'latvians', u'saudi arabian',
      u'saint-martinois', u'norfolk island', u'north korean', u'belarusians', u'catrachos', u'tunisians', u'motswana', u'philippine', u'scotswomen',
       u'ecuadorian', u'angolans', u'mongolian', u'abkhazians', u'trinibagonians', u'tobagonian', u'scotsmen', u'salvadoran', u'russians', u'samoan',
        u'llanitos', u'south korean', u'zimbabwean', u'equatoguineans', u'barth\xe9lemoises', u'mauritian', u'sri lankan', u'british virgin island',
         u'emirians', u'bissau-guinean', u'ukrainians', u'statian', u'qatari', u'sint maarten', u'germans', u'estonian', u'cuban', u'slovene',
          u'cura\xe7aoans', u'papuans', u'banyarwanda', u'liechtenstein', u'greenlanders', u'sint maartener', u'moldovans', u'see taiwan',
           u'hong kongese', u'pitcairn islanders', u'malian', u'german', u'austrians', u'moroccan', u'jordanian', u'cambodian', u'puerto ricans',
            u'senegalese', u'singaporean', u'dutchmen', u'mozambican', u'romanian', u'saints', u'filipinos', u'bruneian', u'namibians', u'anguillan',
             u'caymanians', u'montserratians', u'nevisians', u'hungarian', u'comorian', u'emiri', u'tajikistani', u'malawian', u'niueans', u'nevisian',
              u'turks and caicos islanders', u'emiratis', u'tuvaluans', u'malaysian', u'\xe5land island', u'comoran', u'angolan', u'uzbeks', u'scots',
               u'papuan', u'kazakhs', u'philippinean', u'channel islander', u'cook island', u'tuvaluan', u'american samoan', u'nicas', u'taiwanese',
                u'cura\xe7aoan', u'welshmen', u'guambat', u'martiniquais', u'swedes', u'europeans', u'indonesians', u'israeli', u'turks', u'guyanese',
                 u'northern irishmen', u'mexicans', u'salvadorans', u'puerto rican', u'welshwomen', u'colombian', u'niuean', u'belizean', u'persians',
                  u'fijians', u'bolivian', u'sahrawian', u'liberian', u'armenian', u'slovenes', u'bulgarians', u'filipinas', u'new caledonian', u'manx',
                   u'cypriots', u'bahraini', u'montenegrins', u'martinican', u'canadians', u'algerians', u'libyan', u'jamaicans', u'i-kiribati',
                    u'scottish', u'togolese', u'swazi', u'barth\xe9lemois', u'saudis', u'seychellois', u'rwandans', u'bamar', u'american samoans',
                     u'laos', u'guadeloupians', u'saba', u'swazis', u'miquelonnaises', u'cabo verdeans', u'new caledonians', u'charr\xfaas', u'somali',
                      u'grenadians', u'emirati', u'zambian', u'venezuelans', u'sammarinese', u'kyrgyzstani', u'croatians', u'lebanese', u'zambians',
                       u'frenchwomen', u'bruneians', u'saint vincentian', u'see other words for british', u'uzbek', u'herzegovinians', u'ivorians',
                        u'zimbabweans', u'kosovan', u'ivorian', u'cook islanders', u'hondurans', u'nicaraguans', u'danes', u'bahrainis', u'saint-pierraises',
u'saint vincentians', u'sudanese', u'kuwaitis', u'kosovar', u'cariocas', u'bonaire dutch', u'saint-pierrais', u'norfolk islander', u'magyar',
 u'paraguayans', u'comorians', u'azerbaijani', u'french polynesians', u'tajikistanis', u'ghanaians', u'northern irishwomen',
  u'south sandwich islands', u'singapore', u'welsh', u'frenchmen', u'djiboutians', u'guineans', u'sahrawi', u'afghan', u'kazakh',
   u'montserratian', u'mahoran', u'saint lucian', u'central africans', u'iraqis', u'basotho', u'liberians', u'english', u'swiss',
    u'micronesians', u'the', u'kazakhstanis', u'andorran', u'motswana (sing. batswana)', u'bermudans', u'liechtensteiners', u'belarusian',
     u'macedonian', u'bangladeshis', u'falkland islanders', u'cocos islanders', u'polish', u'cameroonian', u'icelandic', u'serbs',
      u'saba dutch', u'ugandan', u'solomon island', u'ethiopians', u'r\xe9unionese', u'falkland island', u'ascension and tristan da cunha',
       u'wallisian', u'channel island', u'indonesian', u'kyrgyzstanis', u'antiguans', u'vincentian', u'ni-vanuatu', u'tongans', u'finnish',
        u'bosnians', u'u.s. virgin island', u'serbian', u'bermudan', u'pinoys', u'italian', u'portuguese', u'bonaire', u'chadians',
         u'quisqueyanos', u'czech', u'finns', u'barbadians', u'kittitians', u'laotian', u'sierra leoneans', u'bajans', u'new zealand',
          u'uruguayans', u'marshallese', u'libyans', u'papua new guinean', u'jan mayen', u'ukrainian', u'saint helenians', u'luxembourg',
           u'malinese', u'jordanians', u'englishwomen', u'british virgin islanders', u'tobagonians', u'trinidadian', u'englishmen',
            u'hungarians', u'paraguayan', u'romanians', u'saint-martinoises', u'kazakhstani', u'chilean', u'vatican citizens', u'gambian',
             u'kosovars', u'hellenic', u'albanian', u'hongkies', u'maubere', u'antiguan', u'latvian', u'peruvians', u'palestinians',
              u'trinidadians', u'boricuas', u'united states', u'slovak', u'u.s. virgin islanders', u'beninese', u'pitcairn island',
               u'cocos island', u'serbians', u'grenadian', u'bangladeshi', u'biot', u'brazilian', u'venezuelan', u'estonians',
                u'cameroonians', u'azeri', u'south georgia', u'bosnian', u'bermudians', u'hongers', u'south ossetians', u'turkmen',
                 u'russian', u'haitian', u'wallisians', u'statians', u'dominican', u'macedonians', u'somalians', u'us', u'below',
                  u'persian', u'jamaican', u'uk', u'vatican', u'saint lucians', u'beninoises', u'irish', u'cypriot', u'peruvian',
                   u'guamanians', u'singaporeans', u'yemeni', u'croats', u'syrian', u'surinamers', u'south african', u'nicaraguan',
                    u'guinean', u'ticos', u'djiboutian', u'armenians', u'seychelloises', u'nz', u'turkmens', u'nigeriens', u'rwandan',
                     u'netherlandic', u'kenyan', u'irishmen', u'tokelauan', u'uzbekistanis', u'aussies', u'equatoguinean', u'monacan', u'nauruans',
                      u'wallis and futuna', u'nepali', u'australians', u'ecuadorians', u'heard island', u'norwegians', u'mahorans',
                       u'lithuanians', u'panamanians'])


class transition_model_discrim:
	trans_data = []
	feature_index = {}
	label_index = {}
	custom_regions = []
	country_names = []
	region_bin_dict = {}

	def __init__(self):
		self.trans_counts = {}

	def load(self, direct):
		import ParseLGL
		conn = psycopg2.connect(os.environ['DB_CONN'])

		self.load_custom_regions()
		self.load_country_names()
		self.load_region_bin_dict()

		cur = conn.cursor()
		m = 0
		for xml_infile in os.listdir(direct):
			
			print xml_infile
			m += 1
			wordref, toporef, domain = ParseLGL.parse_xml(os.path.join(direct, xml_infile))
			self.trans_data = featurize_transition_discrim(wordref, toporef, domain, cur, self.country_names)

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

	#feature_set should be output of discrim_featurize function	
	def log_prob_dict(self, feature_set):
		feature_vector = np.zeros(len(self.feature_index))
		

		prob_dict = {}
		if len(self.label_index) > 2:
			for label in self.label_index:
				label_sum = 0.0
				label_sum += self.weights[self.label_index[label]][0]
				for feat in feature_set:
					if feat in self.feature_index:
						label_sum += self.weights[self.label_index[label]][self.feature_index[feat]]
				prob_dict[label] = math.log(math.exp(label_sum) / (1.0 + math.exp(label_sum)))
			return prob_dict
		elif len(self.label_index) == 2:
			discount = .10
			inv_index = {v: k for k, v in self.label_index.items()}
			label = inv_index[1]
			label_sum = 0.0
			label_sum += self.weights[0][0]
			for feat in feature_set:
				if feat in self.feature_index:
					label_sum += self.weights[0][self.feature_index[feat]]
			prob_dict[label] = math.exp(label_sum) / (1.0 + math.exp(label_sum))
			label2 = inv_index[0]
			prob_dict[label2] = (1.0 - (math.exp(label_sum) / (1.0 + math.exp(label_sum))))
			if "CONTINENT/GLOBAL" not in self.label_index:
				prob_dict2 = {}
				prob_dict2["CONTINENT/GLOBAL"] = math.log(sum([prob_dict[l]*discount for l in prob_dict]))
				for pb in prob_dict:
					prob_dict2[pb] = math.log(prob_dict[pb] - (prob_dict[pb]*discount))
				#print "HAVING TO BACK OFF TO CONTINENT/GLOBAL INTERP"
				#prob_dict2[pb]
				#print feature_set
				#sys.exit()
				return prob_dict2
			return prob_dict


	def load_region_bin_dict(self):
		conn = psycopg2.connect(os.environ['DB_CONN'])
		cur = conn.cursor()
		region_bin_dict = {}
		print "Loading Region Bin Dict"
		for region in self.custom_regions:
			SQL = "SELECT p1.region_name, p2.region_name, ST_DWithin(p1.geog, p2.geog, 161000.0)  from customgrid as p1, customgrid as p2 where p1.region_name = %s;"
			cur.execute(SQL, (region, ))
			results = cur.fetchall()

			results.append([region, region, True])

			for row in results:
				#print row
				reg1 = row[0]
				reg2 = row[1]
				if reg1 == reg2:
					if reg1 not in region_bin_dict:
						region_bin_dict[reg1] = {}
					region_bin_dict[reg1][reg2] = 'SAME'
				elif row[2] == True:
					if reg1 not in region_bin_dict:
						region_bin_dict[reg1] = {}
					region_bin_dict[reg1][reg2] = "LOCAL/ADJACENT"
				else:
					if reg1 not in region_bin_dict:
						region_bin_dict[reg1] = {}
					region_bin_dict[reg1][reg2] = "CONTINENT/GLOBAL"
		self.region_bin_dict = region_bin_dict
		conn.close()					

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

def discrim_featurize(prev_toponame, cur_toponame, token_dist, country_names):
	obs_features = []
	#Token_Bins = {'adjacent':[0, 4], 'sentence':[5, 25], 'paragraph':[26, 150], 'document':[151, 4000]}

	#Add the duplicate toponym feature
	if prev_toponame.lower() == cur_toponame.lower():
		obs_features.append('SAME_TOPO')

	#Add the token distance bin feature
	#tokebin = get_tokenbin(Token_Bins, token_dist)
	#if tokebin in Token_Bins:
	#	obs_features.append(tokebin)

	#'County' is in the toponym
	#if 'county' in cur_toponame.lower():
	#	obs_features.append('CUR_COUNTY')
	#if 'county' in prev_toponame.lower():
	#	obs_features.append('PREV_COUNTY')

	#Add the demonym features
	if len(isDemonym(prev_toponame, demonyms)) > 0:
		obs_features.append('PREV_DEMONYM')
	if len(isDemonym(cur_toponame, demonyms)) > 0:
		obs_features.append('CUR_DEMONYM')

	#Add the is a country name feature
	if len(isCountryName(prev_toponame, country_names)) > 0:
		obs_features.append('PREV_COUNTRYNAME')
	if len(isCountryName(cur_toponame, country_names)) > 0:
		obs_features.append('CUR_COUNTRYNAME')


	return obs_features


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


def viterbi_discrim(obs, states, TM, LM, cur):
	V = [{}]
	path = {}

	# Initialize base cases (t == 0)
	#print obs[0][-1]
	emission_dict = get_emission_dict(LM, obs[0][0])
	for y in states:
		V[0][y] = emission_dict[y]
		path[y] = [y]

	# Run Viterbi for t > 0
	for t in range(1, len(obs)):
		V.append({})
		newpath = {}
		emission_dict = get_emission_dict(LM, obs[t][0])
		transition_probdict = TM.log_prob_dict(obs[t][1])
		print obs[t]
		print transition_probdict
		for y in states:
			#print emission_dict[y]
			#print math.log(emission_dict[y])
			#    print math.log(TM.binomial_prob(j, y))
			(prob, state) = max((V[t-1][y0] + transition_probdict[TM.region_bin_dict[y0][y]] + emission_dict[y], y0) for y0 in states)
			#(prob, state) = max((V[t-1][y0] + math.log(TM.binomial_prob(y0, y)) + emission_dict[y], y0) for y0 in states)
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
	#SQL_DIST  = "SELECT ST_DWithin(p1.geog, p2.geog, 1500000.0) from customgrid as p1, customgrid as p2 where p1.region_name = %s and p2.region_name = %s;"
	#cur.execute(SQL_DIST, (current_region, prev_region))
	#results = cur.fetchall()
	#if results[0][0] == True:
	#	return "COUNTRY"
	return "CONTINENT/GLOBAL"

#RETURN COUNTRY if toponym is a name or alt name of a country
def isCountryName(toponym, country_names):
	if toponym.lower() in country_names:
		return 'COUNTRY'
	else:
		return ''

#RETURN Demonym is toponym is a demonym
def isDemonym(toponym, demonyms):
	if toponym.lower() in demonyms:
		return 'Demonym'
	else:
		return ''


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
			
			#print last_topo[0], "->", toporef[i][0]
			
			token_dist = i - last_topo[-1]

			label = getRegionBin(current_region, prev_region, cur)

			obs_features = discrim_featurize(prev_toponame, toponym, token_dist, country_names)

			transition_data.append([label, obs_features])
			#print transition_data[-1]

			
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

def test_viterbi_discrim_poly(LM, TM, directory="/work/02608/grantdel/corpora/LGL/articles/dev_testsplit1", poly_table_name = "lgl_dev_classic"):
	import ParseLGL

	out_test = "test_output3.txt"

	ot = io.open(out_test, 'w', encoding='utf-8')

	conn = psycopg2.connect(os.environ['DB_CONN'])
	cur = conn.cursor()

	cor = 0
	total = 0
	for f in os.listdir(directory):
		obs_sequence = []
		#print f
		wordref, toporef, domain = ParseLGL.parse_xml(os.path.join(directory, f))
		topo_context_dict = ParseLGL.getTopoContexts(wordref, toporef, window=1)
		ordered_tkeys = sorted(topo_context_dict.keys())
		obs = [[topo, topo_context_dict[topo]['entry'][0], topo_context_dict[topo]['context'].keys()] for topo in ordered_tkeys]
		#print obs
		#print "==="
		j = 0
		for o in obs:
			j += 1
			topo = o[1]
			topo_tokeid = o[0]
			if j > 1:
				toke_dist = topo_tokeid - prev_topo_tokeid 	
				trans_features = discrim_featurize(prev_topo, topo, toke_dist, TM.country_names)
				obs_sequence.append([o[2], trans_features])
			else:
				obs_sequence.append([o[2], []])
			prev_topo = topo
			prev_topo_tokeid = o[0]
		#print "obs"
		#print obs
		#print "==="
		states = TM.custom_regions
		if len(obs_sequence) > 0:
			prob, prob_path = viterbi_discrim(obs_sequence, states, TM, LM, cur)
			zipped_preds = zip(prob_path, [toporef[topo] for topo in ordered_tkeys])
			print "prob path", zipped_preds

			for pred in zipped_preds:
				pred_region = pred[0]
				lat = float(pred[1][1]['lat'])
				lon = float(pred[1][1]['long'])

				did = pred[1][-2]
				wid = pred[1][-1]
				#print "pred:", pred_region
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

	print "VITERBI DISCRIM POLY ACC:"
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
	for f in os.listdir(directory):
		obs_sequence = []
		#print f
		wordref, toporef, domain = ParseLGL.parse_xml(os.path.join(directory, f))
		topo_context_dict = ParseLGL.getTopoContexts(wordref, toporef, window=1)
		ordered_tkeys = sorted(topo_context_dict.keys())
		obs = [[topo, topo_context_dict[topo]['entry'][0], topo_context_dict[topo]['context'].keys()] for topo in ordered_tkeys]
		#print obs
		#print "==="
		j = 0
		for o in obs:
			j += 1
			topo = o[1]
			topo_tokeid = o[0]
			if j > 1:
				toke_dist = topo_tokeid - prev_topo_tokeid 	
				trans_features = discrim_featurize(prev_topo, topo, toke_dist, TM.country_names)
				obs_sequence.append([o[2], trans_features])
			else:
				obs_sequence.append([o[2], []])
			prev_topo = topo
			prev_topo_tokeid = o[0]
		#print "obs"
		#print obs
		#print "==="
		states = TM.custom_regions
		if len(obs_sequence) > 0:
			prob, prob_path = viterbi_discrim(obs_sequence, states, TM, LM, cur)
			zipped_preds = zip(prob_path, [toporef[topo] for topo in ordered_tkeys])
			print "prob path", zipped_preds

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

	print "VITERBI DISCRIM ACC:"
	print cor, "/", total
	print float(cor)/float(total)

	ot.close()
	conn.close()

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
TM.load("/work/02608/grantdel/corpora/LGL/articles/dev_trainsplit1")
TM.train()
#test_pureLM(LM, directory="/work/02608/grantdel/corpora/trconllf/dev_testsplit5")
test_viterbi_discrim(LM, TM, directory="/work/02608/grantdel/corpora/LGL/articles/dev_testsplit1")

#test_pureLM_poly(LM, directory="/work/02608/grantdel/corpora/LGL/articles/dev_testsplit4", poly_table_name="lgl_dev_classic")
#test_viterbi_poly(LM, TM, directory="/work/02608/grantdel/corpora/LGL/articles/dev_testsplit4", poly_table_name="lgl_dev_classic")
#test_viterbi_discrim_poly(LM, TM, directory="/work/02608/grantdel/corpora/LGL/articles/dev_testsplit1", poly_table_name="lgl_dev_classic")



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
