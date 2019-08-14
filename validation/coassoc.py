import itertools
import logging as log
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------

class CoassociationMatrix:

	def __init__( self ):
		self.num_topic_models = 0
		self.all_terms = set()
		self.pair_counts = {}

	def add( self, topic_model ):
		self.num_topic_models += 1
		# process each topic
		for ranking in topic_model:
			# add to set of all terms
			for term in ranking:
				self.all_terms.add(term)
			# increment pair counts
			for term1, term2 in itertools.combinations(ranking, 2):
				pair = frozenset([term1,term2])
				if pair in self.pair_counts:
					self.pair_counts[pair] += 1
				else:
					self.pair_counts[pair] = 1

	def build_matrix( self ):
		log.info("Building co-association matrix for %d terms from %d topic models" % ( len(self.all_terms), self.num_topic_models ) )
		# by default, order the terms alphabetically
		terms = list(self.all_terms)
		terms.sort()
		num_terms = len(terms)
		# create a dictionary to map terms to indices, to improve performance
		term_map = {}
		for i, term in enumerate( terms ):
			term_map[term] = i
		# populate the matrix
		C = np.zeros( (num_terms,num_terms) )
		for pair in self.pair_counts:
			term1, term2 = pair
			i, j = term_map[term1], term_map[term2]
			C[i,j] = float(self.pair_counts[pair])/self.num_topic_models
			C[j,i] = C[i,j]
		# set the diagonal to be the maximum co-association value 1
		np.fill_diagonal(C, 1)
		return C, terms

# --------------------------------------------------------------------------------------

class TermClusterer:

	def __init__( self, k = 2, max_topic_terms = 10 ):
		self.k = k
		self.max_topic_terms = max_topic_terms

	def find_clusters( self, dfw ):
		# build up to K clusters
		term_clusters = []
		assigned_terms = set()
		for i in range( self.k ):
			log.debug("Building cluster %d/%d" % ((i+1), self.k ) )
			seed = self.find_next_seed( dfw, assigned_terms )
			cluster = self.expand_seed( dfw, seed )
			term_clusters.append( cluster )
			# update the current set of expanded terms
			assigned_terms = assigned_terms.union( cluster )
		return term_clusters

	def find_next_seed( self, dfw, assigned_terms ):
		terms = list(dfw.index)
		max_value = 0
		max_pair = []
		for pair in itertools.combinations(terms, r=2):
			if pair[0] in assigned_terms or pair[1] in assigned_terms:
				continue
			if dfw.loc[pair[0],pair[1]] > max_value:
				max_value = dfw.loc[pair[0],pair[1]]
				max_pair = pair
		log.debug("Best unassigned pair: %s = %.3f" % ( max_pair, max_value) )
		return max_pair

	def expand_seed( self, dfw, seed ):
		"""
		Expand a given seed cluster, up the maximum number of terms
		"""
		terms = list(dfw.index)
		# what is the actual maximum number of terms per cluster?
		actual_max_terms = min( self.max_topic_terms, len(dfw) )
		# try to expand the cluster
		cluster = list( seed )
		while len( cluster ) < actual_max_terms:
			# find the next best term
			max_score = -1
			best_term = None
			for candidate in terms:
				if candidate in cluster:
					continue
				score = 0.0
				for term in cluster:
					score += dfw.loc[candidate,term]
				score /= len(cluster)
				if score > max_score:
					max_score = score
					best_term = candidate
			if best_term is None:
				break
			cluster.append( best_term )
		log.debug(cluster)
		return cluster	

# --------------------------------------------------------------------------------------

class WeightedCoassociation:

	def __init__( self, embedding, top_terms ):
		self.embedding = embedding
		self.top_terms = top_terms
		self.co = CoassociationMatrix()
		self.k = 2
		self.term_clusters = None

	def add( self, topic_model ):
		self.co.add( topic_model )
		# use the maximum value of k from any of the base models
		self.k = max( self.k, len(topic_model) )

	def build_embedding_similarity_matrix( self, terms ):
		# create a dictionary to map terms to indices, to improve performance
		term_map = {}
		for i, term in enumerate( terms ):
			term_map[term] = i
		# populate the matrix
		S = np.zeros( ( len(terms),len(terms) ) )
		for term1, term2 in itertools.combinations(terms, 2):
			i, j = term_map[term1], term_map[term2]
			try:
				S[i,j] = self.embedding.similarity( term1, term2 )
				S[j,i] = S[i,j]
			except:
				# ignore missing terms
				pass
		# set the diagonal to be the maximum similarity value
		np.fill_diagonal(S, 1)    
		return S

	def evaluate( self  ):
		# Make sure we have performed the clustering step
		if self.term_clusters is None:
			self.apply()
		# Return the overall evaluate is the mean of the term cluster coherence scores
		return self.cluster_coherences.mean()

	def apply( self ):
		# Step 1. Create an unweighted co-association matrix
		self.C, all_terms = self.co.build_matrix()
		self.dfc = pd.DataFrame( self.C, index=all_terms, columns=all_terms )
		# Step 2. Create a pairwise term similarity matrix, based on the embedding:
		self.S = self.build_embedding_similarity_matrix( all_terms  )
		self.dfs = pd.DataFrame( self.S, index=all_terms, columns=all_terms )
		# Step 3. Create the weighted co-association matrix, which considers both co-associations and term similarities
		self.W = np.multiply(self.C, self.S)
		self.dfw = pd.DataFrame( self.W, index=all_terms, columns=all_terms )		
		# Step 4. Apply clustering
		self.term_clusters = TermClusterer( self.k, self.top_terms ).find_clusters( self.dfw )
		# Step 5. Evaluate the term clusters
		self.cluster_coherences = []
		self.cluster_pair_coherences = []
		for i, term_cluster in enumerate(self.term_clusters):
			cluster_coherence, pair_coherences = self.evaluate_term_cluster( self.dfw, term_cluster )
			self.cluster_coherences.append( cluster_coherence )
			self.cluster_pair_coherences.append( pair_coherences )
		self.cluster_coherences = np.array( self.cluster_coherences )

	def evaluate_term_cluster( self, dfw, term_cluster ):
		cluster_coherence = 0
		pair_coherences = {}
		# process each pair of terms
		for term1, term2 in itertools.combinations(term_cluster, 2):
			sim = dfw.loc[term1,term2]
			cluster_coherence += sim
			# add the pair score
			pair_coherences[ frozenset([term1, term2]) ] = sim
		# apply denominators
		cluster_coherence /= len(pair_coherences)
		return ( cluster_coherence, pair_coherences )

