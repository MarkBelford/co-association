#!/usr/bin/env python
"""
Tool to pre-process documents in the specified directories, and export a single pre-processed datasets, ready for topic modeling. 

python prep-text.py --tfidf --norm -o data/sample data/sample/ 
"""
import os, os.path, sys
import logging as log
from optparse import OptionParser
import text.util

# --------------------------------------------------------------

def main():
	parser = OptionParser(usage="usage: %prog [options] directory1 directory2 ...")
	parser.add_option("--df", action="store", type="int", dest="min_df", help="minimum number of documents for a term to appear", default=10)
	parser.add_option("--tfidf", action="store_true", dest="apply_tfidf", help="apply TF-IDF term weight to the document-term matrix")
	parser.add_option("--norm", action="store_true", dest="apply_norm", help="apply unit length normalization to the document-term matrix")
	parser.add_option("--minlen", action="store", type="int", dest="min_doc_length", help="minimum document length (in characters)", default=10)
	parser.add_option("-s", action="store", type="string", dest="stoplist_file", help="generic stopword file path", default="text/stopwords.txt")
	parser.add_option("-o", action="store", type="string", dest="prefix", help="output prefix for corpus files", default="corpus")
	# Parse command line arguments
	(options, args) = parser.parse_args()
	if( len(args) < 1 ):
		parser.error( "Must specify at least one directory" )	
	log.basicConfig(level=20, format='%(message)s')

	# Load required stopwords
	log.info( "Using stopwords from %s" % options.stoplist_file ) 
	stopwords = text.util.load_word_set( options.stoplist_file )
	log.info( "%d stopwords loaded" % len(stopwords) )

	# Read content of all documents in the specified directories
	docgen = text.util.DocumentBodyGenerator( args, options.min_doc_length )
	docs = []
	doc_ids = []
	classes, label_count = {}, {}
	for doc_id, filepath, body in docgen:
		label = os.path.basename( os.path.dirname( filepath ).replace(" ", "_") )
		if label not in classes:
			classes[label] = set()
			label_count[label] = 0
		classes[label].add(doc_id)
		label_count[label] += 1
		docs.append(body)	
		doc_ids.append(doc_id)	
	log.info( "Found %d documents to parse" % len(docs) )
	if len(classes) < 2:
		log.warning( "No ground truth available" )
		classes = None
	else:
		log.info( "Ground truth: %d classes - %s" % ( len(classes), label_count ) )

	# Convert the documents in TF-IDF vectors and filter stopwords
	log.info( "Pre-processing data (%d stopwords, tfidf=%s, normalize=%s, min_df=%d) ..." % (len(stopwords), options.apply_tfidf, options.apply_norm, options.min_df) )
	(X,terms) = text.util.preprocess( docs, stopwords, min_df = options.min_df, apply_tfidf = options.apply_tfidf, apply_norm = options.apply_norm )
	log.info( "Built document-term matrix: %d documents, %d terms" % (X.shape[0], X.shape[1]) )
	
	# Store the corpus
	prefix = options.prefix
	log.info( "Saving data to %s.pkl" % prefix )
	text.util.save_corpus( prefix, X, terms, doc_ids, classes )

# --------------------------------------------------------------

if __name__ == "__main__":
	main()
