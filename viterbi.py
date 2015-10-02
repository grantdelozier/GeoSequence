import math


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

def viterbi_discrim_tagdict(obs, states, TM, LM, cur):
	V = [{}]
	path = {}

	# Initialize base cases (t == 0)
	#print obs[0][-1]
	emission_dict = get_emission_dict(LM, obs[0][0])
	for y in states:
		if y in obs[0][2] and 'CUR_DEMONYM' not in obs[0][1]:
			V[0][y] = emission_dict[y]
			path[y] = [y]
		elif len(obs[0][2]) == 0 or 'CUR_DEMONYM' in obs[0][1]:
			V[0][y] = emission_dict[y]
			path[y] = [y]

	# Run Viterbi for t > 0
	for t in range(1, len(obs)):
		V.append({})
		newpath = {}
		emission_dict = get_emission_dict(LM, obs[t][0])
		transition_probdict = TM.log_prob_dict(obs[t][1])
		print obs[t]
		#print transition_probdict
		for y in states:
			#print emission_dict[y]
			#print math.log(emission_dict[y])
			#    print math.log(TM.binomial_prob(j, y))
			if y in obs[t][2] and 'CUR_DEMONYM' not in obs[t][1]:
				if len(obs[t-1][2]) == 0 or 'CUR_DEMONYM' in obs[t-1][1]:
					(prob, state) = max((V[t-1][y0] + transition_probdict[TM.region_bin_dict[y0][y]] + emission_dict[y], y0) for y0 in states)
				else:
					(prob, state) = max((V[t-1][y0] + transition_probdict[TM.region_bin_dict[y0][y]] + emission_dict[y], y0) for y0 in states if y0 in obs[t-1][2])
				#(prob, state) = max((V[t-1][y0] + math.log(TM.binomial_prob(y0, y)) + emission_dict[y], y0) for y0 in states)
				V[t][y] = prob
				newpath[y] = path[state] + [y]
			elif len(obs[t][2]) == 0 or 'CUR_DEMONYM' in obs[t][1]:
				if len(obs[t-1][2]) == 0 or 'CUR_DEMONYM' in obs[t-1][1]:
					(prob, state) = max((V[t-1][y0] + transition_probdict[TM.region_bin_dict[y0][y]] + emission_dict[y], y0) for y0 in states)
				else:
					(prob, state) = max((V[t-1][y0] + transition_probdict[TM.region_bin_dict[y0][y]] + emission_dict[y], y0) for y0 in states if y0 in obs[t-1][2])
				#(prob, state) = max((V[t-1][y0] + math.log(TM.binomial_prob(y0, y)) + emission_dict[y], y0) for y0 in states)
				V[t][y] = prob
				newpath[y] = path[state] + [y]

		# Don't need to remember the old paths
		path = newpath
	n = 0           # if only one element is observed max is sought in the initialization values
	if len(obs) != 1:
		n = t
	#print_dptable(V)
	if len(obs[n][2]) == 0 or 'CUR_DEMONYM' in obs[n][1]:
		(prob, state) = max((V[n][y], y) for y in states)
	else:
		(prob, state) = max((V[n][y], y) for y in states if y in obs[n][2])
	return (prob, path[state])


def print_dptable(V):
    s = "    " + " ".join(("%7d" % i) for i in range(len(V))) + "\n"
    for y in V[0]:
        s += "%.5s: " % y
        s += " ".join("%.7s" % ("%f" % v[y]) for v in V)
        s += "\n"
    print(s)