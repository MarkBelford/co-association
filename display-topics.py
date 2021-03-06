#!/usr/bin/env python
"""
Simple tool to display term rankings generated by NMF, stored in one or more PKL files.

Sample usage:
python display-topics.py -t 10 data/bbc/nmf_k05/*rank*

"""
import logging as log
from optparse import OptionParser
import unsupervised.rankings, unsupervised.util

# --------------------------------------------------------------

def main():
	parser = OptionParser(usage="usage: %prog [options] ranking_file1 ranking_file2 ...")
	parser.add_option("-t", "--top", action="store", type="int", dest="top", help="number of top terms to show", default=10)
	parser.add_option("-l","--long", action="store_true", dest="long_display", help="long format display")
	(options, args) = parser.parse_args()
	if( len(args) < 1 ):
		parser.error( "Must specify at least one ranking set file" )
	log.basicConfig(level=20, format='%(message)s')
	
	# Load each cached ranking set
	for in_path in args:
		log.info( "Loading terms from %s ..." % in_path )
		(term_rankings,labels) = unsupervised.util.load_term_rankings( in_path )
		m = unsupervised.rankings.term_rankings_size( term_rankings )
		actual_top = min(options.top,m)
		log.info( "Set has %d rankings covering up to %d terms" % ( len(term_rankings), m ) )
		tab = unsupervised.rankings.DescriptorTable( term_rankings, actual_top )
		# topics on rows or columns?
		if options.long_display:
			log.info( tab.format_long() )
		else:
			log.info( tab.format() )

# --------------------------------------------------------------

if __name__ == "__main__":
	main()
