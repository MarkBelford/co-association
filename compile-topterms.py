#!/usr/bin/env python
"""
Utility tool to list all unique top terms from one or more topic models.

Sample usage:
python compile-topterms.py -t 10 -o sample-top10.txt results/
"""
import os, os.path, sys, io
import logging as log
from optparse import OptionParser
import unsupervised.rankings, unsupervised.util

# --------------------------------------------------------------

def find_model_filepaths( in_paths ):
	filepaths = []
	for root_path in in_paths:
		for dir_path, subFolders, files in os.walk(root_path):
			for filename in files:
				if filename.startswith("ranks") and filename.endswith(".pkl"):
					filepath = os.path.join(dir_path,filename)
					filepaths.append( filepath )
	filepaths.sort()
	return filepaths	

# --------------------------------------------------------------

def main():
	parser = OptionParser(usage="usage: %prog [options] directory1 directory2 ...")
	parser.add_option("-t", "--top", action="store", type="int", dest="top", help="number of top terms to include for each topic", default=10)
	parser.add_option("-o", action="store", type="string", dest="out_path", help="output path for list of terms (by default print to screen)", default=None)
	(options, args) = parser.parse_args()
	if( len(args) < 1 ):
		parser.error( "Must specify at least one directory" )	
	top = options.top
	log.basicConfig(level=20, format='%(message)s')

	# Get list of all specified term ranking files
	file_paths = find_model_filepaths( args )
	if len(file_paths) == 0:
		log.error("No term ranking files found to validate")
		sys.exit(1)
	log.info( "Found %d models to parse" % len(file_paths) )

	# process each topic model results file
	terms = set()
	num_models = 0 
	for in_path in file_paths:
		log.debug("Processing %s ...." % in_path)
		(term_rankings,labels) = unsupervised.util.load_term_rankings( in_path )
		if options.top > 1:
			term_rankings = unsupervised.rankings.truncate_term_rankings( term_rankings, options.top )		
		num_models += 1
		for ranking in term_rankings:
			for term in ranking:
				terms.add(term)

	terms = list(terms)
	terms.sort()
	log.info("Found %d unique terms when considering top %d rankings" % ( len(terms), options.top ) )

	if options.out_path is None:
		for term in terms:
			log.info(term)
	else:
		log.info( "Writing %d terms from %d models to %s ..." % ( len(terms), num_models, options.out_path ) )
		fout = io.open(options.out_path, 'w', encoding="utf8", errors='ignore')
		for term in terms:
			fout.write("%s\n" % term )
		fout.close()

# --------------------------------------------------------------

if __name__ == "__main__":
	main()		
