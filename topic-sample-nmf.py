#!/usr/bin/env python
"""
Applies randomly-initialized NMF to the specified dataset to generate a collection of topic models, 
using a randomly sampled fraction of the dataset for each run.

Sample usage:
python topic-sample-nmf.py sample.pkl -s 0.8 --kmin 5 --kmax 5 -r 20 --seed 1000 --maxiters 100 -o models/sample
"""
import os, sys
import logging as log
from optparse import OptionParser
import numpy as np
import scipy.sparse
import textutil, unsupervised.nmf, unsupervised.rankings, unsupervised.util

# --------------------------------------------------------------

def main():
	parser = OptionParser(usage="usage: %prog [options] dataset_file")
	parser.add_option("--seed", action="store", type="int", dest="seed", help="random seed", default=1000)
	parser.add_option("--kmin", action="store", type="int", dest="kmin", help="minimum number of topics", default=5)
	parser.add_option("--kmax", action="store", type="int", dest="kmax", help="maximum number of topics", default=5)
	parser.add_option('--step' ,type="int", dest="step", help="Step size for incrementing the number of topics", default=1 )
	parser.add_option("-r","--runs", action="store", type="int", dest="runs", help="number of runs", default=1)
	parser.add_option("-s", "--sample", action="store", type="float", dest="sample_ratio", help="sampling ratio of documents to include in each run (range is 0 to 1). default=0.8", default=0.8)
	parser.add_option("--maxiters", action="store", type="int", dest="maxiter", help="maximum number of iterations", default=100)
	parser.add_option("-o","--outdir", action="store", type="string", dest="dir_out", help="base output directory (default is current directory)", default=None)
	parser.add_option('-d','--debug',type="int",help="Level of log output; 0 is less, 5 is all", default=3)
	# Parse command line arguments
	(options, args) = parser.parse_args()
	if len(args) < 1:
		parser.error( "Must specify at least one dataset file" )	
	if options.sample_ratio < 0 or options.sample_ratio > 1:
		parser.error( "Must specify a sampling ratio in the range [0,1]" )	
	log_level = max(50 - (options.debug * 10), 10)
	log.basicConfig(level=log_level, format='%(message)s')
	init_strategy = "random"

	if options.dir_out is None:
		dir_out_base = os.getcwd()
	else:
		dir_out_base = options.dir_out

	# Load the cached corpus
	corpus_path = args[0]
	(X,terms,doc_ids,classes) = textutil.load_corpus( corpus_path )
	log.info("Loaded dataset - %d documents, %d terms" % ( len(doc_ids), len(terms) ) )

	# Choose implementation
	impl = unsupervised.nmf.NMFWrapper( max_iters = options.maxiter, init_strategy = init_strategy )

	# Set up sampling
	n_documents = X.shape[0]
	n_sample = int( options.sample_ratio * n_documents )
	indices = np.arange(n_documents)

	# Generate all NMF topic models for the specified numbers of topics
	log.info( "Generating NMF models in range k=[%d,%d], init_strategy=%s" % ( options.kmin, options.kmax, init_strategy ) )
	log.info( "Sampling ratio = %.2f - %d/%d documents per run" % ( options.sample_ratio, n_sample, n_documents ) )
	for k in range(options.kmin, options.kmax+1, options.step):
		# Set random state
		unsupervised.util.init_random_seeds( options.seed )		
		log.info( "Applying NMF with sampling (k=%d, runs=%d, seed=%s) ..." % ( k, options.runs, options.seed ) )
		dir_out_k = os.path.join( dir_out_base, "sample_k%02d" % k )
		if not os.path.exists(dir_out_k):
			os.makedirs(dir_out_k)		
		log.debug( "Results will be written to %s" % dir_out_k )

		# Run NMF
		for r in range(options.runs):
			log.info( "NMF run %d/%d (k=%d, max_iters=%d)" % (r+1, options.runs, k, options.maxiter ) )
			sample_rate_per = int( 100 * options.sample_ratio )
			file_suffix = "sample%03d_%s_%03d" % ( sample_rate_per, options.seed, r+1 )
			# randomly subsample the data
			log.debug("Subsamping the data ...")
			np.random.shuffle(indices)
			sample_indices = indices[0:n_sample]


			S = X[sample_indices,:]
			log.debug("Creating sparse matrix ...")
			S = scipy.sparse.csr_matrix(S)

			sample_doc_ids = []
			for doc_index in sample_indices:
				sample_doc_ids.append( doc_ids[doc_index] )		

			# apply NMF
			impl.apply( S, k )

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

			# Write sampled document partition
			partition = impl.generate_partition()
			partition_out_path = os.path.join( dir_out_k, "partition_%s.pkl" % file_suffix )
			log.debug( "Writing document partition for %d documents to %s"% ( len(sample_doc_ids), partition_out_path ) )
			unsupervised.util.save_partition( partition_out_path, partition, sample_doc_ids )

			# Write full document partition
			full_partition = np.repeat(-1, n_documents)
			for i, doc_index in enumerate(sample_indices):
				full_partition[doc_index] = partition[i]
			
			full_partition_out_path = os.path.join( dir_out_k, "full_partition_%s.pkl" % file_suffix )
			log.debug( "Writing document partition for %d documents to %s"% ( len(doc_ids), full_partition_out_path ) )
			unsupervised.util.save_partition( full_partition_out_path, full_partition, doc_ids )

			# Write the factorization
			factor_out_path = os.path.join( dir_out_k, "factors_%s.pkl" % file_suffix )
			# NB: need to make a copy of the factors
			log.debug( "Writing factorization for %d documents to %s" % ( len(sample_doc_ids), factor_out_path ) )
			unsupervised.util.save_nmf_factors( factor_out_path, np.array( impl.W ), np.array( impl.H ), sample_doc_ids, terms )
		  
# --------------------------------------------------------------

if __name__ == "__main__":
	main()
