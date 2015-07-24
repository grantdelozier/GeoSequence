import os
import xml.etree.ElementTree as ET
import psycopg2

#
def parse_xml(afile):
	xmldoc = ET.parse(file(afile))
	root = xmldoc.getroot()

	wordref = {}
	toporef = {}
	i = 0
	sid = 0
	domain = ""
	
	for child in root.iter('doc'):
		did = child.attrib['id']
		if 'domain' in child.attrib:
			domain = child.attrib['domain']
		else: domain = "NA"
		sid = 0
		for c in child:
			sid += 1
			wid = 0
			#print sid
			for sub in c:
				#print sub.tag, sub.attrib
				if sub.tag == "w" and len(sub.attrib['tok']) > 0:
					i += 1
					#print sub.attrib['tok']
					wordref[i] = sub.attrib['tok']
					wid += 1
				elif sub.tag == "toponym":
					i += 1
					#print sub.attrib['term']
					wordref[i] = sub.attrib['term']
					wid += 1
					for sub2 in sub:
						for sub3 in sub2:
							if "selected" in sub3.attrib:
								#print sub3.attrib
								toporef[i] = [wordref[i], sub3.attrib, did, wid]
	return wordref, toporef, domain


pronouns = set([u"you", u"you'd", u"you'll", u"you're", u"you've", u"your", u"yours", u"yourself", 
			u"yourselves", u"we", u"we'd", u"we'll", u"we're", u"we've", u"it", u"it's",
			u"its", u"itself", u"i", u"i'd", u"i'll", u"i'm", u"i've", u"he", u"he'd", 
			u"he'll", u"he's", u"her", u"hers", u"herself", u"him", u"himself", u"his",
			u"our", u"ours", u"us"])

def find_ngrams(input_list, n):
 	return zip(*[input_list[i:] for i in range(n)])

def is_pronoun(s):
	if s in pronouns:
		return "#PRO#"
	else:
		return s

def is_number(s):
	try:
		float(s)
		return "#NUMBER#"
	except ValueError:
		if "," in s:
			ss = s.split(",")
			if ss[0].isdigit() == True or ss[1].isdigit() == True:
				return "#NUMBER#"
			else: return s
		else: return s

def is_grammar(s):
	marks = [u",", u".", u"-", u";", u":", u"--", u')', u'(', u'"', u"\u2014", u"\u2018", u"\u2019", u"\u201d", u"\u201c", u"\u2013"]
	if s in marks:
		return "#MARK#"
	else:
		return s

def getTopoContexts(wordref, toporef, window=10):
	topo_context_dict = {}
	for t in toporef:
		contextlist = []
		i = t - window
		if i <= 0:
			i = 1
		while i < t:
			if " " in wordref[i]:
				for w in wordref[i].split(" "):
					contextlist.append(w)
			else:
				contextlist.append(wordref[i])
			i += 1
		if " " in wordref[t]:
			for w in wordref[t].split(" "):
				contextlist.append(w)
		else:
			contextlist.append(wordref[t])
		i = t + 1
		while i <= t+window and i < len(wordref):
			if " " in wordref[i]:
				for w in wordref[i].split(" "):
					contextlist.append(w)
			else:
				contextlist.append(wordref[i])
			i += 1
		unigramlist = [is_pronoun(is_grammar(is_number(w))) for w in contextlist]
		d = {}
		bigrams = [item[0]+'|'+item[1] for item in find_ngrams(unigramlist, 2)]
		for i in unigramlist:
			d[i] = d.get(i, 0.0) + (1.0)
		for i in bigrams:
			d[i] = d.get(i, 0.0) + (1.0)
		topo_context_dict[t] = {'entry':toporef[t], 'context':d}

	return topo_context_dict



'''LGL_Dir = "/home/grant/devel/TopCluster/LGL/articles/dev_classicxml"
for f in os.listdir(LGL_Dir):
	print f
	wordref, toporef, domain = parse_xml(os.path.join(LGL_Dir, f))
	topo_context_dict = getTopoContexts(wordref, toporef)
	print topo_context_dict'''