import itertools
from collections import Counter
import logging as log
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------

class TermCoassociationMatrix:

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
		log.info("Building term co-association matrix for %d terms from %d topic models" % ( len(self.all_terms), self.num_topic_models ) )
		# by default, order the terms alphabetically
		terms = sorted(list(self.all_terms))
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
