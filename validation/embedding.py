import numpy as np
import logging as log
import itertools

class EmbeddingCoherence:
	"""
	Uses an embediding model (e.g. Word2Vec model) to evaluate the coherence of the
	term rankings from a topic model.
	"""
	def __init__( self, model ):
		self.model = model

	def evaluate( self, term_rankings):
		overall = 0.0
		for ranking in term_rankings:
			overall += self.evaluate_ranking( ranking ) 
		return overall / len(term_rankings)

	def evaluate_ranking( self, ranking ):    
		total, pairs = 0.0, 0
		for term1, term2 in itertools.combinations(ranking, 2):
			try:
				total += self.model.similarity( term1, term2 )
				pairs += 1
			except Exception as e:
				log.info( e )
		if pairs == 0:
			return 0.0
		return total/pairs


class EmbeddingDistinctiveness:
	"""
	Uses an embediding model (e.g. Word2Vec model) to evaluate the distinctiveness of the
	term rankings from a topic model.
	"""
	def __init__( self, model ):
		self.model = model

	def evaluate( self, term_rankings ):
		total_sim, pairs = 0.0, 0
		max_scores = []
		for i in range(len(term_rankings)):
			max_similarity = 0.0
			for j in range(len(term_rankings)):
				if i == j:
					continue
				else:
					similarity = self.evaluate_similarity(term_rankings[i], term_rankings[j] )
				if similarity > max_similarity:
					max_similarity = similarity
			max_scores.append(max_similarity)
#		log.info("Found %d values" % len(max_scores))
#		log.info("Average Max Similarity: %f" % np.mean(max_scores))
		return 1.0 - np.mean(max_scores)

	def evaluate_similarity( self, ranking1, ranking2 ):
		total, pairs = 0.0, 0
		for term1 in ranking1:
			for term2 in ranking2:
				try:
					total += self.model.similarity( term1, term2 )
					pairs += 1
				except Exception as e:
					log.info( e )
		if pairs == 0:
			return 0.0
		return total/pairs
