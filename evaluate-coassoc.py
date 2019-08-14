#!/usr/bin/env python
"""
Tool for evaluating weighted co-association scores for topic models.

Sample usage, for a single embedding model:
python evaluate-assoc.py -t 10 -m embeddings/model-w2v.bin -o results/embedding-k03.csv models/ranks*.pkl

Sample usage, for a directory containing multiple embedding models:
python evaluate-assoc.py -t 10 -m embeddings/ -o results/embedding-k03.csv models/ranks*.pkl
"""
import os, os.path, sys, csv
import logging as log
from optparse import OptionParser
import gensim
from prettytable import PrettyTable
import unsupervised.rankings, unsupervised.util
import validation.coassoc, validation.util

# --------------------------------------------------------------

def main():
	parser = OptionParser(usage="usage: %prog [options] ranking_file1 ranking_file2 ...")
	parser.add_option("-m", "--model", action="store", type="string", dest="model_path", help="path to an embedding file or directory containing embeddings", default=None)
	parser.add_option("-t", "--top", action="store", type="int", dest="top", help="number of top terms to include for each topic", default=10)
	parser.add_option("-o","--output", action="store", type="string", dest="out_path", help="path for CSV output file", default=None)
	(options, args) = parser.parse_args()
	if len(args) < 1 :
		parser.error( "Must specify at least one topic model ranking file" )	
	if options.model_path is None:
		parser.error( "Must specify path to embedding file")
	log.basicConfig(level=20, format='%(message)s')
	top = options.top

	if os.path.isdir( options.model_path ):
		model_paths = []
		for fname in os.listdir( options.model_path ):
			if fname.endswith(".bin"):
				model_paths.append( os.path.join( options.model_path, fname ) )
	else:
		model_paths = [ options.model_path ]

	# Load the embedding models and create the measures
	measures = {}
	for model_path in model_paths:
		if "-ft" in model_path:
			log.info("Loading FastText model from %s ..." % model_path )
			model = gensim.models.FastText.load(model_path)
			vocab = set(model.wv.vocab.keys() )
		else:
			log.info("Loading Word2vec model from %s ..." % model_path )
			model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)	
			vocab = set(model.vocab.keys())
		log.info("Embedding has vocabulary of size %d" % len(vocab) )
		# create the measure
		suffix = os.path.splitext( os.path.split( model_path )[-1] )[0]
		# TODO: revert?
		measures[ "wcoassoc2" + suffix ] = validation.coassoc.WeightedCoassociation( model, top )

	# Process each topic model results file
	log.info( "Processing %d topic models ..." % len(args) )
	for in_path in args:
		log.debug("Processing topics from %s using top %d terms" % ( in_path, top ) )
		(term_rankings,labels) = unsupervised.util.load_term_rankings( in_path )
		log.debug("Truncating terms rankings to top %d terms" % top )
		truncated_rankings = unsupervised.rankings.truncate_term_rankings( term_rankings, top, vocab )
		for key in measures:
			measures[key].add( truncated_rankings )

	# Display a summary of the results
	tab = PrettyTable( ["experiment", "score"] )
	tab.align["experiment"] = "l"
	for key in measures:
		score = measures[key].evaluate()
		tab.add_row( [ key, "%.3f" % score ] ) 
	log.info(tab)

	# Write results to CSV?
	if not options.out_path is None:
		log.info("Writing results to %s" % options.out_path)
		fout = open(options.out_path, 'w')
		w = csv.writer( fout, delimiter=",", quoting=csv.QUOTE_MINIMAL)
		w.writerow(tab.field_names)
		for row in tab._rows:
			w.writerow(row)
		fout.close()

# --------------------------------------------------------------

if __name__ == "__main__":
	main()		
