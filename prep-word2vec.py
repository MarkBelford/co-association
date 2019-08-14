#!/usr/bin/env python
"""
Tool to pre-process documents in the specified directories, and build a Word2Vec word embedding model. 

This implementation requires Gensim. For documentation regarding the various parameters, see:
https://radimrehurek.com/gensim/models/word2vec.html

Same usage:
python prep-word2vec.py --df 10 -m sg -d 100 -o data/model-w2v.bin data/sample/
"""
import os, os.path, sys
import logging as log
from optparse import OptionParser
import gensim
import text.util

# --------------------------------------------------------------

def main():
	parser = OptionParser(usage="usage: %prog [options] directory1 directory2 ...")
	parser.add_option("--df", action="store", type="int", dest="min_df", help="minimum number of documents for a term to appear", default=10)
	parser.add_option("-d","--dimensions", action="store", type="int", dest="dimensions", help="the dimensionality of the word vectors", default=500)
	parser.add_option("--window", action="store", type="int", dest="w2v_window", help="the maximum distance for Word2Vec to use between the current and predicted word within a sentence", default=5)
	parser.add_option("-m", action="store", type="string", dest="model_type", help="type of word embedding model to build (sg or cbow)", default="sg")
	parser.add_option("-s", action="store", type="string", dest="stoplist_file", help="custom stopword file path", default="text/stopwords.txt")
	parser.add_option("--minlen", action="store", type="int", dest="min_doc_length", help="minimum document length (in characters)", default=10)
	parser.add_option("-b", "--binary", action="store_true", dest="write_binary", help="write a Word2Vec file in binary format")
	parser.add_option("-o", action="store", type="string", dest="out_path", help="output path for corpus files", default="model-w2v.bin")
	# Parse command line arguments
	(options, args) = parser.parse_args()
	if( len(args) < 1 ):
		parser.error( "Must specify at least one directory" )	
	log.basicConfig(level=20, format='%(message)s')

	# Load required stopwords
	log.info( "Using stopwords from %s" % options.stoplist_file )
	stopwords = text.util.load_word_set( options.stoplist_file )

	# Process all specified directories
	docgen = text.util.DocumentTokenGenerator( args, options.min_doc_length, stopwords )

	# Build the Word2Vec model from the documents that we have found
	log.info( "Building Word2vec %s model..." % options.model_type )
	if options.model_type == "cbow":
		model = gensim.models.Word2Vec(docgen, size=options.dimensions, min_count=options.min_df, window=options.w2v_window, workers=4, sg = 0)
	elif options.model_type == "sg":
		model = gensim.models.Word2Vec(docgen, size=options.dimensions, min_count=options.min_df, window=options.w2v_window, workers=4, sg = 1)
	else:
		log.error("Unknown model type '%s'" % options.model_type )
		sys.exit(1)
	log.info( "Built model: %s" % model )

	# Save the Word2Vec model
	log.info( "Writing model to %s ..." % options.out_path )
	if options.write_binary:
		model.wv.save_word2vec_format(options.out_path, binary=True) 
	else:
		model.save(options.out_path)
			
# --------------------------------------------------------------

if __name__ == "__main__":
	main()
