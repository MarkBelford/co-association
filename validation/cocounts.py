import itertools, scipy, math
import logging as log
from scipy.sparse import dok_matrix

# --------------------------------------------------------------

def get_cocount_index(x,y):
    # Upper triangular matrix
    return (x,y) if x < y else (y,x)

def term_ranking_to_indexes( ranking, term_map ):
    topic_indexes = []
    for term in ranking:
        if term in term_map:
            topic_indexes.append( term_map[term] )
    return topic_indexes

class CocountMeasure:
	def __init__( self, cocounts, term_map, total_windows, epsilon = 1.0 ):
		self.cocounts = cocounts
		self.term_map = term_map
		self.total_windows = total_windows
		# As suggested by Mimno et al. 2011 (and used by Stevens et al. 2012), this is a smoothing factor to avoid 
		# taking the logarithm of 0, i.e. when two terms never co-occur in the external corpus
		self.epsilon = epsilon

	def evaluate( self, term_rankings ):
		overall = 0.0
		for ranking in term_rankings:
			overall += self.evaluate_ranking( ranking ) 
		return overall / len(term_rankings)

	def evaluate_ranking( self, ranking ):    
		return self.evaluate_index_coherence( term_ranking_to_indexes( ranking, self.term_map ) )

	def evaluate_index_coherence( self, term_indexes ):
		total, pairs = 0.0, 0
		for x, y in itertools.combinations(term_indexes, 2):
			total += self.evaluate_pair( x, y )
			pairs += 1
		return total/pairs

	def evaluate_pair( self, x, y ):
		pass

# --------------------------------------------------------------

def calc_pmi( count_x, count_y, combined_count, total_windows, epsilon ):
	if count_x == 0 or count_y == 0 or combined_count == 0:
		return 0.0
	numer = combined_count * total_windows
	denom = count_x * count_y
	denom = denom if denom > 0.0 else 1.0/total_windows
	pmi = math.log((numer + epsilon)/denom, 10)
	return pmi

class PMICoherence(CocountMeasure):
	def __init__( self, cocounts, term_map, total_windows ):
		super(PMICoherence, self).__init__(cocounts, term_map, total_windows)

	def evaluate_pair( self, x, y ):
		count_x = self.cocounts[get_cocount_index(x,x)]
		count_y = self.cocounts[get_cocount_index(y,y)]
		combined_count = self.cocounts[get_cocount_index(x,y)]
		return calc_pmi( count_x, count_y, combined_count, self.total_windows, self.epsilon )

class NPMICoherence(CocountMeasure):
	def __init__( self, cocounts, term_map, total_windows ):
		super(NPMICoherence, self).__init__(cocounts, term_map, total_windows)

	def evaluate_pair( self, x, y ):
		count_x = self.cocounts[get_cocount_index(x,x)]
		count_y = self.cocounts[get_cocount_index(y,y)]
		combined_count = self.cocounts[get_cocount_index(x,y)]
		pmi = calc_pmi( count_x, count_y, combined_count, self.total_windows, self.epsilon )
		npmi = pmi / (-1.0*math.log((float(combined_count)+ self.epsilon)/(self.total_windows),10))
		return npmi

# --------------------------------------------------------------

def initialize_cocounts_matrix(num_required_terms):
	return dok_matrix((num_required_terms, num_required_terms), dtype=scipy.float32)

def sliding_window(seq, window_size):
	it = iter(seq)
	res = tuple(itertools.islice(it,window_size))
	if len(res) == window_size:
		yield res
	for elem in it:
		res = res[1:] + (elem,)
		yield res

def update_cocounts_matrix(term_cocounts, current_terms):
	sorted_terms = sorted(current_terms)
	num_pairs = 0
	for x,y in itertools.combinations_with_replacement(sorted_terms, 2):
		# As done with Lau, only increment the count if both terms appeared in at least 1 top N topic terms list, i.e. topic_word_rel
		# Index will always be sorted at this point, i.e. upper triangular matrix will be populated
		term_cocounts[x,y] += 1.0
		num_pairs += 1
	return num_pairs

def process_document( current_terms, term_cocounts, window_size ):
	num_terms = len(current_terms)
	num_windows = 0
	num_pairs = 0
	for window_term_map in [set([x for x in current_terms[i:i+window_size] if x > -1]) for i in range(0,num_terms) if i==0 or i+window_size <= num_terms]:
		num_pairs += update_cocounts_matrix(term_cocounts, window_term_map)
		num_windows += 1
	log.debug('EOL for %d terms, window size %d: Found %d windows, %d pairs. Overall matrix now has %d entries' % (len(current_terms), window_size, num_windows, num_pairs, term_cocounts.nnz))
	return num_windows

def calculate_sliding_window_cocounts(generator, window_size, term_map ):
	total_num_terms = len(term_map)
	term_cocounts = initialize_cocounts_matrix(total_num_terms)
	total_windows = 0
	
	for doc_tokens in generator:
		# Placeholder of -1 for non-required terms (that should still be used for the sliding window)
		doc_indices = [term_map.get(x, -1) for x in doc_tokens]
		doc_num_windows = process_document(doc_indices, term_cocounts, window_size)
		total_windows += doc_num_windows
		if generator.num_documents % 5000 == 0 :
			log.info('Current: %d documents. Matrix has %d entries' % ( generator.num_documents, term_cocounts.nnz))
	log.info('Total: %d documents, %d windows. Matrix has %d entries' % ( generator.num_documents, total_windows, term_cocounts.nnz))
	# Note: convert to dense matrix
	return term_cocounts.todense(), total_windows


def calculate_umass_cocounts(generator, term_map):
	term_cocounts = initialize_cocounts_matrix(total_num_terms)
	total_num_terms = len(term_map)

	for doc_tokens in generator:
		doc_tokens = sorted(set(doc_tokens))
		doc_tokens = doc_tokens.intersection(term_map.keys()) # not all tokens in a document are in the term map
		for x,y in itertools.combinations_with_replacement(doc_tokens, 2):
			term_cocounts[term_map[x], term_map[y]] += 1.0

	return term_cocounts.todense()			
# --------------------------------------------------------------
