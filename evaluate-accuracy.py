#!/usr/bin/env python
import os, os.path, sys
import logging as log
from optparse import OptionParser
import numpy as np
import sklearn.metrics.cluster
import textutil, unsupervised.util
import validation.util

""" 
Evaluates the accuracy of the produced document-topic partitions stored in one or more PKL files, with respect to 
	the ground truth labels of a dataset.

	Sample usage:
	python evaluate-accuracy.py -o results/bbc-accuracy.csv data/bbc.pkl data/bbc/nmf_k05/*partition* 
"""

# --------------------------------------------------------------

def validate( measure, classes, clustering ):
	if measure == "nmi":
		return sklearn.metrics.cluster.normalized_mutual_info_score( classes, clustering, average_method='geometric' )
	elif measure == "ami":
		return sklearn.metrics.cluster.adjusted_mutual_info_score( classes, clustering )
	elif measure == "ari":
		return sklearn.metrics.cluster.adjusted_rand_score( classes, clustering )
	if measure == "vm":
		return sklearn.metrics.cluster.v_measure_score( classes, clustering )
	log.error("Unknown validation measure: %s" % measure )
	return None

# --------------------------------------------------------------

def main():
	parser = OptionParser(usage="usage: %prog [options] corpus_file partition_file1|directory1 ...")
	parser.add_option("-o","--output", action="store", type="string", dest="out_path", help="path for CSV output file", default=None)
	parser.add_option("-m", "--measures", action="store_true", dest="measures", help="comma-separated list of validation measures to use (default is nmi)", default="nmi" )
	# Parse command line arguments
	(options, args) = parser.parse_args()
	if( len(args) < 2 ):
		parser.error( "Must specify at least a corpus and one or more partitions/directories" )	
	log.basicConfig(level=20, format='%(message)s')
	measures = [ x.strip() for x in options.measures.lower().split(",") ]

	log.info ("Reading corpus from %s ..." % args[0] )
	# Load the cached corpus
	(X,terms,doc_ids,classes) = textutil.load_corpus( args[0] )
	if classes is None:
		log.error( "Error: No class information available for this corpus")
		sys.exit(1)

	# Convert a map to a list of class indices
	classes_partition = unsupervised.util.clustermap_to_partition( classes, doc_ids )

	# Get list of all specified partition files
	file_paths = []
	for path in args[1:]:
		if not os.path.exists( path ):
			log.error("No such file or directory: %s" % path )
			sys.exit(1)
		if os.path.isdir(path):
			log.debug("Searching %s for partitions" % path )
			for dir_path, dirs, files in os.walk(path):
				for fname in files:
					if fname.startswith("partition") and fname.endswith(".pkl"):
						file_paths.append( os.path.join( dir_path, fname ) )
		else:
			file_paths.append( path )
	file_paths.sort()

	if len(file_paths) == 0:
		log.error("No partition files found to validate")
		sys.exit(1)

	# Validation each partition
	log.info("Processing partitions for %d topic models  ..." % len(file_paths) )
	scores = validation.util.ScoreCollection()
	for file_path in file_paths:
		log.debug( "Loading partition from %s" % file_path )
		partition, cluster_doc_ids = unsupervised.util.load_partition( file_path )
		# does the number of documents match up?
		if len(doc_ids) != len(cluster_doc_ids):
			log.warning("Error: Cannot compare clusterings on different data")
			continue
		# perform validation
		partition_scores = {}
		for measure in measures:
			partition_scores[measure] = validate(measure, classes_partition, partition)
		scores.add( file_path, partition_scores )

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
