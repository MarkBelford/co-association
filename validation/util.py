import csv
import numpy as np
from prettytable import PrettyTable

# --------------------------------------------------------------

class ScoreCollection:
	"""
	A utility class for keeping track of experiment scores produced by multiple validation measures 
	applied to different topic models.
	"""
	def __init__( self ):
		self.all_scores = {}
		self.all_score_keys = set()

	def add( self, experiment_key, scores ):
		for score_key in scores:
			self.all_score_keys.add( score_key )
		self.all_scores[experiment_key] = scores

	def aggregate_scores( self ):
		if len(self.all_scores) == 0:
			return []
		vectors = {}
		for score_key in self.all_score_keys:
			vectors[score_key] = []
		for experiment_key in self.all_scores:
			for score_key in self.all_scores[experiment_key]:
				vectors[score_key].append( self.all_scores[experiment_key][score_key] )
		mean_scores = {}
		std_scores = {}
		for score_key in self.all_score_keys:
			v = np.array( vectors[score_key] )
			mean_scores[score_key] = np.mean(v)
			std_scores[score_key] = np.std(v)
		return (mean_scores,std_scores)

	def create_table( self, include_mean = False, precision = 2 ):
		fmt = "%%.%df" % precision
		header = ["experiment"]
		score_keys = list(self.all_score_keys)
		score_keys.sort()
		header += score_keys
		tab = PrettyTable( header )
		tab.align["experiment"] = "l"
		experiment_keys = list( self.all_scores.keys() )
		experiment_keys.sort()
		for experiment_key in experiment_keys:
			row = [ experiment_key ]
			for score_key in score_keys:
				row.append( fmt % self.all_scores[experiment_key].get( score_key, 0.0 ) )
			tab.add_row( row )
		if include_mean:
			mean_scores, std_scores = self.aggregate_scores()
			row = [ "MEAN" ]
			for score_key in score_keys:
				row.append( fmt % mean_scores.get( score_key, 0.0 ) )
			tab.add_row( row )
		return tab 

	def write_table( self, out_path, delimiter=",", include_mean = False, precision = 2 ):
		tab = self.create_table( include_mean, precision )
		fout = open(out_path, 'w')
		w = csv.writer( fout, delimiter=delimiter, quoting=csv.QUOTE_MINIMAL)
		w.writerow(tab.field_names)
		for row in tab._rows:
			w.writerow(row)
		fout.close()

class CoherenceScoreCollection( ScoreCollection ):
	def __init__( self, measures ):
		ScoreCollection.__init__(self)
		self.measures = measures

	def evaluate( self, experiment_key, term_rankings ):
		experiment_scores = {}
		for measure_name in self.measures:
			experiment_scores[measure_name] = self.measures[measure_name].evaluate( term_rankings )
		self.add( experiment_key, experiment_scores )

