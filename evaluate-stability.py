#!/usr/bin/env python
"""
Tool for evaluating the stability of topic models.

Sample usage:
python evaluate-stability.py -t 10 results/reference-nmf/nmf_k02/ranks_reference.pkl results/topic-nmf/nmf_k02/ranks*
"""
import os, sys
import logging as log
from optparse import OptionParser
import numpy as np
import unsupervised.util, unsupervised.rankings
import validation.util

# --------------------------------------------------------------

def main():
	parser = OptionParser(usage="usage: %prog [options] reference_rank_file test_rank_file1 test_rank_file2 ...")
	parser.add_option("-t", "--top", action="store", type="int", dest="top", help="number of top terms to use", default=20)
	parser.add_option("-o","--output", action="store", type="string", dest="out_path", help="path for CSV output file", default=None)
	parser.add_option('-d','--debug',type="int",help="Level of log output; 0 is less, 5 is all", default=3)
	(options, args) = parser.parse_args()
	if( len(args) < 2 ):
		parser.error( "Must specify at least two ranking sets" )
	log_level = max(50 - (options.debug * 10), 10)
	log.basicConfig(level=log_level, format='%(message)s')

	# Load cached ranking sets
	log.info( "Reading %d term ranking sets (top=%d) ..." % ( len(args), options.top ) )
	all_term_rankings = []
	for rank_path in args:
		# first set is the reference set
		if len(all_term_rankings) == 0:
			log.debug( "Loading reference term ranking set from %s ..." % rank_path )
		else:
			log.debug( "Loading test term ranking set from %s ..." % rank_path )
		(term_rankings,labels) = unsupervised.util.load_term_rankings( rank_path )
		log.debug( "Set has %d rankings covering %d terms" % ( len(term_rankings), unsupervised.rankings.term_rankings_size( term_rankings ) ) )
		# do we need to truncate the number of terms in the ranking?
		if options.top > 1:
			term_rankings = unsupervised.rankings.truncate_term_rankings( term_rankings, options.top )
			log.debug( "Truncated to %d -> set now has %d rankings covering %d terms" % ( options.top, len(term_rankings), unsupervised.rankings.term_rankings_size( term_rankings ) ) )
		all_term_rankings.append( term_rankings )

	# First argument was the reference term ranking
	reference_term_ranking = all_term_rankings[0]
	all_term_rankings = all_term_rankings[1:]
	r = len(all_term_rankings)
	log.info( "Loaded %d non-reference term rankings" % r )

	# Perform the evaluation
	metric = unsupervised.rankings.JaccardBinary()
	matcher = unsupervised.rankings.RankingSetAgreement( metric )	
	log.info( "Evaluating stability %d base term rankings with %s and top %d terms ..." % (r, str(metric), options.top))
	scores = validation.util.ScoreCollection()

	#all_scores = []
	#for i in range(r):
	#	for j in range(i+1, r):
	#		agreement = matcher.similarity(all_term_rankings[i], all_term_rankings[j] )
	#		all_scores.append(agreement)
	#log.info("Calculted score based on %d pairs" % len(all_scores))
	#scores.add( args[i+1], { "agreement" : np.mean(all_scores) } )

	for i in range(r):
		agreement = matcher.similarity(all_term_rankings[i], reference_term_ranking)
		scores.add( args[i+1], { "agreement" : agreement } )

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
