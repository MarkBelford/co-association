import numpy as np
import pandas as pd
from tabulate import tabulate
import unsupervised.hungarian

# --------------------------------------------------------------
# Ranking Similarity 
# --------------------------------------------------------------

class JaccardBinary:
	""" 
	Simple binary Jaccard-based ranking comparison, which does not take into account rank positions. 
	"""
	def similarity( self, gold_ranking, test_ranking ):
		sx = set(gold_ranking)
		sy = set(test_ranking)
		numer = len( sx.intersection(sy) )
		if numer == 0:
			return 0.0
		denom = len( sx.union(sy) )
		if denom == 0:
			return 0.0
		return float(numer)/denom

	def __str__( self ):
		return "%s" % ( self.__class__.__name__ )

class AverageJaccard(JaccardBinary):
	""" 
	A top-weighted version of Jaccard, which takes into account rank positions. 
	This is based on Fagin's Average Overlap Intersection Metric.
	"""
	def similarity( self, gold_ranking, test_ranking ):
		k = min( len(gold_ranking), len(test_ranking) )
		total = 0.0
		for i in range(1,k+1):
			total += JaccardBinary.similarity( self, gold_ranking[0:i], test_ranking[0:i] )
		return total/k

# --------------------------------------------------------------
# Ranking Set Agreement
# --------------------------------------------------------------

class RankingSetAgreement:
	"""
	Calculates the agreement between pairs of ranking sets, using a specified measure of 
	similarity between rankings.
	"""
	def __init__( self, metric = AverageJaccard() ):
		self.metric = metric

	def similarity( self, rankings1, rankings2 ):
		"""
		Calculate the overall agreement between two different ranking sets. This is given by the
		mean similarity values for all matched pairs.
		"""
		self.results = None
		self.S = self.build_matrix( rankings1, rankings2 )
		score, self.results = self.hungarian_matching()
		return score

	def build_matrix( self, rankings1, rankings2 ):
		"""
		Construct the similarity matrix between the pairs of rankings in two 
		different ranking sets.
		"""
		rows = len(rankings1)
		cols = len(rankings2)
		S = np.zeros( (rows,cols) )
		for row in range(rows):
			for col in range(cols):
				S[row,col] = self.metric.similarity( rankings1[row], rankings2[col] )
		return S	

	def hungarian_matching( self ):
		"""
		Solve the Hungarian matching problem to find the best matches between columns and rows based on
		values in the specified similarity matrix.
		"""
		# apply hungarian matching
		h = unsupervised.hungarian.Hungarian()
		C = h.make_cost_matrix(self.S)
		h.calculate(C)
		results = h.get_results()
		# compute score based on similarities
		score = 0.0
		for (row,col) in results:
			score += self.S[row,col]
		score /= len(results)
		return (score, results)

# --------------------------------------------------------------
# Topic Display
# --------------------------------------------------------------

class DescriptorTable:

	def __init__( self, term_rankings, top = 10, labels = None ):
		self.term_rankings = term_rankings
		self.top = top
		self.labels = labels
		self.df = self._populate_df()

	def _populate_df( self ):
		k = len(self.term_rankings)
		if self.labels is None:
			columns = []
			for i in range( k ):
				columns.append("C%02d" % (i+1) )
		else:
			columns = list(self.labels)
		rows = []
		for i in range( self.top ):
			row = { "Rank" : i+1 }
			for j in range( k ):
				col_name = columns[j]
				row[col_name] = self.term_rankings[j][i]
			rows.append( row )
		columns.insert( 0, "Rank" )
		return pd.DataFrame( rows, columns=columns ).set_index("Rank")

	def format( self ):
		"""
		Format a list of multiple term rankings, one ranking per column.
		"""
		return tabulate(self.df, headers='keys', tablefmt='psql')

	def format_long( self ):
		"""
		Format a list of multiple term rankings, one ranking per row.
		"""
		# create the long tablee
		rows = []
		long_columns = [ "Topic", "Top %d Terms" % self.top ]
		for col in self.df.columns:
			terms = list(self.df[col])
			sterms = ", ".join( terms )
			row = { "Topic" : col, long_columns[-1] : sterms }
			rows.append( row )
		long_df = pd.DataFrame( rows, columns=long_columns ).set_index("Topic")
		# format the table
		return tabulate(long_df, headers='keys', tablefmt='psql')

	def get_df( self ):
		return self.df

	def to_csv( self, out_path, sep="\t" ):
		self.df.to_csv( out_path, sep )

# --------------------------------------------------------------
# Utilities
# --------------------------------------------------------------

def calc_relevance_scores( n, rel_measure ):
	""" 
	Utility function to compute a sequence of relevance scores using the specified function.
	"""
	scores = []
	for i in range(n):
		scores.append( rel_measure.relevance( i + 1 ) )
	return scores

def term_rankings_size( term_rankings ):
	"""
	Return the number of terms covered by a list of multiple term rankings.
	"""
	m = 0
	for ranking in term_rankings:
		if m == 0:
			m = len(ranking)
		else:
			m = min( len(ranking), m ) 
	return m

def truncate_term_rankings( orig_rankings, top, vocab = None ):
	"""
	Truncate a list of multiple term rankings to the specified length, possibly filtered based
	on the specified vocabulary.
	"""
	trunc_rankings = []
	if vocab is None:
		if top < 1:
			return orig_rankings
		for ranking in orig_rankings:
			trunc_rankings.append( ranking[0:min(len(ranking),top)] )
	else:
		total = 0
		for ranking in orig_rankings:
			counter = 0
			temp = []
			for term in ranking:
				if term in vocab:
					temp.append( term )
				else:
					counter += 1
				if len(temp) == top:
					break
			total += counter
			trunc_rankings.append( temp )
		#print('Skipped %d terms' % total)
	return trunc_rankings


