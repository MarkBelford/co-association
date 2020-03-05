#!/usr/bin/env python
"""
Tool for evaluating the coherence of topic models stred in one or more PKL files, using measure based on word embeddings.

Sample usage:
python evaluate-embedding.py -b -t 10 -m wikipedia2016-w2v-cbow-d100.bin -o results/bbc-coherence.csv data/bbc/nmf_k05/*rank*

"""
import os, os.path, sys
import logging as log
from optparse import OptionParser
import gensim
import unsupervised.rankings, unsupervised.util
import validation.embedding, validation.util

# --------------------------------------------------------------

def main():
	parser = OptionParser(usage="usage: %prog [options] ranking_file1 ranking_file2 ...")
	parser.add_option("-m", "--model", action="store", type="string", dest="model_path", help="path to Word2Vec model", default=None)
	parser.add_option("-t", "--top", action="store", type="int", dest="top", help="number of top terms to include for each topic", default=10)
	parser.add_option("-s", "--summmary", action="store_true", dest="summary", help="display summary results only")
	parser.add_option("-o","--output", action="store", type="string", dest="out_path", help="path for CSV output file", default=None)
	parser.add_option("-b", "--binary", action="store_true", dest="read_binary", help="read a Word2Vec file in binary format")
	(options, args) = parser.parse_args()
	if len(args) < 1 :
		parser.error( "Must specify at least one topic model ranking file" )	
	if options.model_path is None:
		parser.error( "Must specify path to Word2Vec model file")
	log.basicConfig(level=20, format='%(message)s')
	top = options.top

	# Load the word2vec model and create the measures
	log.info( "Loading embedding model from %s ..." % args[0] )
	if options.read_binary:
		model = gensim.models.KeyedVectors.load_word2vec_format(options.model_path, binary=True)
	else:
		model = gensim.models.Word2Vec.load(options.model_path) 
	vocab = set(model.vocab.keys())
	log.info("Embedding has vocabulary of size %d" % len(vocab) )

	# Create coherence measures
	measures = { "tc-w2v" : validation.embedding.EmbeddingCoherence(model)}# "td-w2v" : validation.embedding.EmbeddingDistinctiveness(model) }
	scores = validation.util.CoherenceScoreCollection( measures )

	# Process each topic model results file
	log.info( "Processing %d topic models ..." % len(args) )
	for in_path in args:
		log.debug("Processing topics from %s using top %d terms" % ( in_path, top ) )
		(term_rankings,labels) = unsupervised.util.load_term_rankings( in_path )
		log.debug("Truncating terms rankings to top %d terms" % top )
		truncated_rankings = unsupervised.rankings.truncate_term_rankings( term_rankings, top)#, vocab )
		scores.evaluate( in_path, truncated_rankings )

	# Display a summary of the results
	tab = scores.create_table( include_stats = True, precision = 3 )
	log.info(tab)

	# Write results to CSV?
	if not options.out_path is None:
		log.info("Writing results to %s" % options.out_path)
		scores.write_table( options.out_path, include_stats = True, precision = 4 )

# --------------------------------------------------------------

if __name__ == "__main__":
	main()		
