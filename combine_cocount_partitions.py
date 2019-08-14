import os
import numpy as np
import logging as log
from optparse import OptionParser
from sklearn.externals import joblib

"""
Tool to combine multiple cocount partitions into one combined cocount matrix 

Sample Usage:
python combine_cocount_partitions.py -o cocounts/combined-cocounts/bbcsport.pkl cocounts/cocount-partitions/bbcsport/
""" 

def load_cocount_partitions(in_path): # 0 = cocount matrix, 1 = term_map, 2 = inv_map, 3 = window_count
	""" Loads previously calculated cocount matrices and term maps from each reference corpus partition """  
	cocount_partitions = []
	for file in os.listdir(in_path):
		if file.endswith('.pkl'):
			cocount_partitions.append(joblib.load(os.path.join(in_path, file)))
	return cocount_partitions

def generate_term_map(cocount_partitions):
	""" Generates new complete term map as some terms may be missing in certain partitions """
	terms = set()
	for partition in cocount_partitions:
		for term in partition[1].keys(): 
			terms.add(term)
		term_map = dict(zip(sorted(terms), np.arange(0,len(terms) + 1)))
	return term_map

def create_matrix(term_map):
	size = len(term_map)	
	return np.zeros(shape=(size,size))

def combine_partitions(cocount_partitions, new_term_map, combined_cocount_matrix):
	""" Combines all partition cocount matrices into one final combined matrix based on new term mapping """
	num_windows = 0
	for partition in cocount_partitions:
		old_cocount_matrix = partition[0]
		old_inv_term_map = partition[2]
		num_windows += partition[3] # num_windows is needed in the pmi caclculations 
		for x in range(0, old_cocount_matrix.shape[0]):
			x_term = old_inv_term_map[x] # old term map will map an index to a term
			x_new_index = new_term_map[x_term] # then map term to new index 
			for y in range(0, old_cocount_matrix.shape[1]):
				y_term = old_inv_term_map[y]
				y_new_index = new_term_map[y_term]   
				combined_cocount_matrix[x_new_index, y_new_index] += old_cocount_matrix[x,y]
	return combined_cocount_matrix, num_windows

#-----------------------------------------------

def main():
	parser = OptionParser(usage="usage: %prog [options] directory")
	parser.add_option("-o","--output", action="store", type="string", dest="out_path", help="path for combined cocount  pkl output", default=None)
	
	(options, args) = parser.parse_args()

	if len(args) < 1 :
		parser.error( "Must specify folder containing partition pkl files" )

	print("Loading partitions...")
	cocount_partitions = load_cocount_partitions(args[0])
	print("Found %d partitions..." % len(cocount_partitions))
	new_term_map = generate_term_map(cocount_partitions)
	new_inv_map = {v: k for k, v in new_term_map.items()}
	combined_cocount_matrix = create_matrix(new_term_map)
	print("Combining partitions...")
	combined_cocount_matrix, num_windows = combine_partitions(cocount_partitions, new_term_map, combined_cocount_matrix)
	print("Combined %d partitions into a %d * %d  matrix" % ( len(cocount_partitions), combined_cocount_matrix.shape[0], combined_cocount_matrix.shape[1] ))

	cache = [ combined_cocount_matrix, new_term_map, new_inv_map, num_windows]
	joblib.dump(cache, options.out_path)


# --------------------------------------------------------------

if __name__ == "__main__":
	main()

