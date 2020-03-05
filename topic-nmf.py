#!/usr/bin/env python
"""
Applies NMF to the specified dataset to generate a collection of topic models.

Sample usage:
python topic-nmf.py dataset.pkl --init random --kmin 5 --kmax 5 -r 20 --seed 1000 --maxiters 100 -o models/dataset
"""
import os, sys
import logging as log
from optparse import OptionParser
import numpy as np
import textutil, unsupervised.nmf, unsupervised.rankings, unsupervised.util

# --------------------------------------------------------------

def main():
	parser = OptionParser(usage="usage: %prog [options] dataset_file")
	parser.add_option("--seed", action="store", type="int", dest="seed", help="random seed", default=1000)
	parser.add_option("--kmin", action="store", type="int", dest="kmin", help="minimum number of topics", default=5)
	parser.add_option("--kmax", action="store", type="int", dest="kmax", help="maximum number of topics", default=5)
	parser.add_option("-i","--init", action="store", type="string", dest="init_strategy", help="initialization strategy (random or nndsvd)", default="random")
	parser.add_option('--step' ,type="int", dest="step", help="Step size for incrementing the number of topics", default=1 )
	parser.add_option("-r","--runs", action="store", type="int", dest="runs", help="number of runs", default=1)
	parser.add_option("--maxiters", action="store", type="int", dest="maxiter", help="maximum number of iterations", default=100)
	parser.add_option("-o","--outdir", action="store", type="string", dest="dir_out", help="base output directory (default is current directory)", default=None)
	parser.add_option('-d','--debug',type="int",help="Level of log output; 0 is less, 5 is all", default=3)
	# Parse command line arguments
	(options, args) = parser.parse_args()
	if len(args) < 1:
		parser.error( "Must specify at least one dataset file" )	
	log_level = max(50 - (options.debug * 10), 10)
	log.basicConfig(level=log_level, format='%(message)s')

	if options.dir_out is None:
		dir_out_base = os.getcwd()
	else:
		dir_out_base = options.dir_out

	# Load the cached corpus
	corpus_path = args[0]
	(X,terms,doc_ids,classes) = textutil.load_corpus( corpus_path )
	log.info("Loaded dataset - %d documents, %d terms" % ( len(doc_ids), len(terms) ) )

	# Choose implementation
	impl = unsupervised.nmf.NMFWrapper( max_iters = options.maxiter, init_strategy = options.init_strategy )

	# Generate all NMF topic models for the specified numbers of topics
	log.info( "Generating NMF models in range k=[%d,%d], init_strategy=%s" % ( options.kmin, options.kmax, options.init_strategy ) )
	for k in range(options.kmin, options.kmax+1, options.step):
		# Set random state
		unsupervised.util.init_random_seeds( options.seed )
		log.info( "Applying NMF (k=%d, runs=%d, seed=%s) ..." % ( k, options.runs, options.seed ) )
		# choose the appropriate output directory
		if options.init_strategy == "random":
			dir_out_k = os.path.join( dir_out_base, "nmf_k%02d" % k )
		else:
			dir_out_k = os.path.join( dir_out_base, "%s_k%02d" % ( options.init_strategy.lower(), k ) )
		if not os.path.exists(dir_out_k):
			os.makedirs(dir_out_k)		
		log.debug( "Results will be written to %s" % dir_out_k )
		# Run NMF
		for r in range(options.runs):
			log.info( "NMF run %d/%d (k=%d, max_iters=%d)" % (r+1, options.runs, k, options.maxiter ) )
			file_suffix = "%s_%03d" % ( options.seed, r+1 )
			# apply NMF
			impl.apply( X, k )
			# Get term rankings for each topic
			term_rankings = []
			for topic_index in range(k):		
				ranked_term_indices = impl.rank_terms( topic_index )
				term_ranking = [terms[i] for i in ranked_term_indices]
				term_rankings.append(term_ranking)
			log.debug( "Generated ranking set with %d topics covering up to %d terms" % ( len(term_rankings), unsupervised.rankings.term_rankings_size( term_rankings ) ) )
			# Write term rankings
			ranks_out_path = os.path.join( dir_out_k, "ranks_%s.pkl" % file_suffix )
			log.debug( "Writing term ranking set to %s" % ranks_out_path )
			unsupervised.util.save_term_rankings( ranks_out_path, term_rankings )
			# Write document partition
			partition = impl.generate_partition()
			partition_out_path = os.path.join( dir_out_k, "partition_%s.pkl" % file_suffix )
			log.debug( "Writing document partition to %s" % partition_out_path )
			unsupervised.util.save_partition( partition_out_path, partition, doc_ids )			
			# Write the complete factorization
			factor_out_path = os.path.join( dir_out_k, "factors_%s.pkl" % file_suffix )
			# NB: need to make a copy of the factors
			log.debug( "Writing factorization to %s" % factor_out_path )
			unsupervised.util.save_nmf_factors( factor_out_path, np.array( impl.W ), np.array( impl.H ), doc_ids, terms )
		  
# --------------------------------------------------------------

if __name__ == "__main__":
	main()
