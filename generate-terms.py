#!/usr/bin/env python

import os
import gensim
from optparse import OptionParser
import unsupervised.rankings, unsupervised.util

"""
Tool to generate an overall set of unique terms that are present in topic rankings from multiple runs of the same algortihm for different values of k, for use in prep-cocounts.py

Sample usage

python generate-terms.py -m wikiabstracts-201610-cbow100.bin -o cocounts/term-lists/ results/bbcsport
""" 

def main():
	parser = OptionParser(usage="usage: %prog [options] directory")
	parser.add_option("-o", action="store", type="string", dest="out_path", help="output path for all top terms")
	parser.add_option("-m", action="store", type="string", dest="model_path", help="path to binary word2vec model")
	parser.add_option("-t", action="store", type="int", dest="top", help="number of top terms to consider", default=10)
	(options, args) = parser.parse_args()

	terms = set()
	filepaths = []

	if( len(args) < 1 ):
		parser.error( "Must specify at least one directory" )

	print('Loading Word2Vec model...')
	model = gensim.models.KeyedVectors.load_word2vec_format(options.model_path, binary=True)
	vocab = set(model.vocab.keys())

	for path in sorted(os.listdir(args[0])):  # find all sub directories that hold the rankings for each value of k
		if path.startswith('nmf'):
			filepaths.append(os.path.join(args[0], path))

	for i in range(len(filepaths)):
		print("Processing K = %s" % str(i + 2))
		for file in os.listdir(filepaths[i]):
			if file.startswith('rank'): # only consider rankings as partition files are also in thse folders
				(term_rankings,labels) = unsupervised.util.load_term_rankings(os.path.join(filepaths[i], file ))
				term_rankings = unsupervised.rankings.truncate_term_rankings(term_rankings, options.top, vocab) # concstruct descriptors using top n terms that are also in word2vec vocab
				for topic in term_rankings:
					for term in topic:
						terms.add(term)

	print("Total Number of Terms Identified: %s" % len(terms))

	dataset = args[0].split('/')[-1]
	print(dataset)

	f = open('%s/%s.txt' % (options.out_path, dataset), 'w')
	for term in sorted(terms):
		f.write(term+'\n')
	f.close()

#-------------------------------------------------

if __name__ == "__main__":
	main()

