#!/usr/bin/env python
"""
Tool for generating weighted co-association clusterings for topic models.

Sample usage:
python evaluate-assoc.py -t 10 -m embeddings/model-w2v.bin -o wcoassoc-ranks.pkl models/ranks*.pkl
"""
import os, sys, random
import logging as log
from optparse import OptionParser
import numpy as np
import gensim
import unsupervised.rankings, unsupervised.util, validation.coassoc

# --------------------------------------------------------------

def main():
	parser = OptionParser(usage="usage: %prog [options] ranking_file1 ranking_file2 ...")
	parser.add_option("-m", "--model", action="store", type="string", dest="model_path", help="path to an embedding file", default=None)
	parser.add_option("-t", "--top", action="store", type="int", dest="top", help="number of top terms to include for each topic", default=10)
	parser.add_option("-o","--output", action="store", type="string", dest="out_path", help="path for output ranks file", default=None)
	(options, args) = parser.parse_args()
	# Parse command line arguments
	if len(args) < 1 :
		parser.error( "Must specify at least one topic model ranking file" )	
	if options.model_path is None:
		parser.error( "Must specify path to embedding file")
	log.basicConfig(level=20, format='%(message)s')
	model_path = options.model_path
	top = options.top

	if options.out_path is None:
		ranks_out_path = "wcoassoc-ranks.pkl"
	else:
		ranks_out_path = options.out_path

	# Load the embedding model
	if "-ft" in model_path:
		log.info("Loading FastText model from %s ..." % model_path )
		model = gensim.models.FastText.load(model_path)
		vocab = set(model.wv.vocab.keys() )
	else:
		log.info("Loading Word2vec model from %s ..." % model_path )
		model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)	
		vocab = set(model.vocab.keys())
	log.info("Embedding has vocabulary of size %d" % len(vocab) )
	
	# Process each topic model results file
	wco = validation.coassoc.WeightedCoassociation( model, top )
	log.info( "Processing %d topic models ..." % len(args) )
	for in_path in args:
		log.debug("Processing topics from %s using top %d terms" % ( in_path, top ) )
		(term_rankings,labels) = unsupervised.util.load_term_rankings( in_path )
		log.debug("Truncating terms rankings to top %d terms" % top )
		truncated_rankings = unsupervised.rankings.truncate_term_rankings( term_rankings, top, vocab )
		wco.add( truncated_rankings )

	# Apply the process
	log.info("Applying weighted co-association analysis...")
	wco.apply()
	for i, term_cluster in enumerate(wco.term_clusters):
		log.info( term_cluster )
	
	log.debug( "Writing term ranking set to %s" % ranks_out_path )
	unsupervised.util.save_term_rankings( ranks_out_path, wco.term_clusters )		  

# --------------------------------------------------------------

if __name__ == "__main__":
	main()
