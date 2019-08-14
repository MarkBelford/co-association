#!/usr/bin/env python
"""
Tool for evaluating the coherence of topic models, using measures based on PMI.

Sample usage:
python evaluate-cocounts.py -f data/sample-cocounts.pkl -o results-cocounts-k03.csv results/nmf_k03/ranks*.pkl
"""
import gensim
import os, os.path, sys
import logging as log
from optparse import OptionParser
from sklearn.externals import joblib
import unsupervised.rankings, unsupervised.util
import validation.cocounts, validation.util
#from shuffle import permute

# --------------------------------------------------------------

def main():
	parser = OptionParser(usage="usage: %prog [options] ranking_file1 ranking_file2 ...")
	parser.add_option("-f", action="store", type="string", dest="cocount_path", help="path to cached co-count file", default=None)
	parser.add_option("-t", "--top", action="store", type="int", dest="top", help="number of top terms to include for each topic", default=10)
	parser.add_option("-s", "--summmary", action="store_true", dest="summary", help="display summary results only")
	parser.add_option("-o","--output", action="store", type="string", dest="out_path", help="path for CSV output file", default=None)
	parser.add_option("--shuffle", action="store_true", dest="apply_shuffling", help="apply shuffling of topics")
	parser.add_option("-m", action="store", type="string", dest="model_path", help="path for word2vec model", default=None)
	(options, args) = parser.parse_args()
	if len(args) < 1 :
		parser.error( "Must specify at least one topic model ranking file" )	
	if options.cocount_path is None:
		parser.error( "Must specify path to cached co-count file")
	log.basicConfig(level=20, format='%(message)s')
	top = options.top

	log.info("Loading co-count matrix from %s ..." % options.cocount_path )
	(cocounts, term_map, inv_map, total_windows) = joblib.load( options.cocount_path )
	log.info("Read co-counts for %d terms" % len(term_map) )
	measures = { 
		"pmi" : validation.cocounts.PMICoherence(cocounts, term_map, total_windows), 
		"npmi" : validation.cocounts.NPMICoherence(cocounts, term_map, total_windows) }
	scores = validation.util.CoherenceScoreCollection( measures )
									

	if options.model_path is not None:
		print('Loading Word2Vec model...')
		model = gensim.models.KeyedVectors.load_word2vec_format(options.model_path, binary=True)
		vocab = set(model.vocab.keys())
	
	# Process each topic model results file
	log.info( "Processing %d topic models ..." % len(args) )
	for in_path in args:
		log.debug("Processing topics from %s using top %d terms" % ( in_path, top ) )
		(term_rankings,labels) = unsupervised.util.load_term_rankings( in_path )
		term_rankings = unsupervised.rankings.truncate_term_rankings( term_rankings, options.top, vocab )
		#if options.apply_shuffling:
		#	log.info("Shuffling term rankings..." )
		#	term_rankings = permute(term_rankings)
		scores.evaluate( in_path, term_rankings )

	# Display a summary of the results
	tab = scores.create_table( include_mean = True, precision = 3 )
	log.info(tab)

	# Write results to CSV?
	if not options.out_path is None:
		log.info("Writing results to %s" % options.out_path)
		scores.write_table( options.out_path, include_mean = True, precision = 4 )

# --------------------------------------------------------------

if __name__ == "__main__":
	main()		
