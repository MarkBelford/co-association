#!/usr/bin/env python
"""
Tool to calculate co-occurrence counts for a set of relevant terms relative to a background corpus.

Sample usage:
python prep-cocounts.py -t data/sample-top10.txt -o data/sample-cocounts data/sample
"""
import os, os.path, sys
import logging as log
from optparse import OptionParser
from sklearn.externals import joblib
import text.util, validation.cocounts

# --------------------------------------------------------------

def main():
	parser = OptionParser(usage="usage: %prog [options] directory1 directory2")
	parser.add_option("-t", action="store", type="string", dest="termlist_path", help="term list file path", default=None)
	parser.add_option('-w','--windowsize', type="int", dest="window_size", help="sliding window size (default=20)", default=20)
	parser.add_option("--minlen", action="store", type="int", dest="min_doc_length", help="minimum document length (in characters)", default=10)
	parser.add_option("-s", action="store", type="string", dest="stoplist_file", help="custom stopword file path", default="text/stopwords.txt")
	parser.add_option("-o", action="store", type="string", dest="prefix", help="output prefix for corpus files", default="cocounts")
	(options, args) = parser.parse_args()
	if( len(args) < 1 ):
		parser.error( "Must specify at least one directory" )	
	if options.termlist_path is None:
		parser.error( "Must specify term list file path" )	
	log.basicConfig(level=20, format='%(message)s')
	# configurable, but default of 20 was used by Stevens, Lau (although Newman 2010 used 10)
	window_size = options.window_size

	# Read the top terms
	log.info("Reading relevant terms from %s ..." % options.termlist_path )
	terms = text.util.load_word_set( options.termlist_path)
	log.info("Found %d terms" % len(terms))
	# Load required stopwords
	log.info( "Using stopwords from %s" % options.stoplist_file )
	stopwords = text.util.load_word_set( options.stoplist_file )

	# Lau's unigram_list, unigram_rev pair
	term_map = dict([(y,x) for x,y in enumerate(sorted(terms))])
	inv_map = {v: k for k, v in term_map.items()}
	num_terms = len(term_map)

	# Read content of all documents in the specified directories
	generator = text.util.DocumentTokenGenerator( args, options.min_doc_length, stopwords, )

	# perform the actual calculations
	log.info( "Applying sliding window of size %d .." % window_size )
	cocounts, total_windows = validation.cocounts.calculate_sliding_window_cocounts(generator, window_size, term_map)

	# store the output using Joblib
	out_path = "%s.pkl" % options.prefix
	log.info("Writing co-count data to %s.pkl" % options.prefix)
	cache = [ cocounts, term_map, inv_map, total_windows ]
	joblib.dump(cache, out_path ) 

# --------------------------------------------------------------

if __name__ == "__main__":
	main()	
