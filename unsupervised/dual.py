import itertools
from collections import Counter
import logging as log
import pandas as pd
import numpy as np
from unsupervised.coassoc import TermCoassociationMatrix, TermClusterer

# --------------------------------------------------------------------------------------

class DocumentTopicMatrix:

	def __init__( self, num_documents ):
		self.num_topic_models = 0
		self.all_terms = set()
		self.document_terms = []
		self.num_documents = num_documents
		# initialize the counter for the number of times each document appears
		self.doc_counter = np.zeros(num_documents)
		# and the counter for the terms associated with each document
		for j in range(num_documents):
			self.document_terms.append( Counter() )

	def add( self, topic_model, partition ):
		self.num_topic_models += 1
		# process each topic
		for ranking in topic_model:
			# add to set of all terms
			for term in ranking:
				self.all_terms.add(term)
		# process all documents
		for doc_index in range(self.num_documents):
			topic_index = partition[doc_index]
			# has this document been assigned to a topic?
			if topic_index != -1:
				self.doc_counter[doc_index] += 1
				for term in topic_model[topic_index]:
					self.document_terms[doc_index][term] += 1

	def build_matrix( self ):
		log.info("Building rectangular matrix for %d terms from %d topic models" % ( len(self.all_terms), self.num_topic_models ) )
		# by default, order the terms alphabetically
		terms = sorted(list(self.all_terms))
		Z = np.zeros( (self.num_documents, len(terms)) )
		for row in range(self.num_documents):
			for term in self.document_terms[row]:
				col = terms.index(term)
				# normalize count by number of models
				if self.doc_counter[row] > 0:
					Z[row,col] = self.document_terms[row][term] / self.doc_counter[row]
		return Z, terms

# --------------------------------------------------------------------------------------

class WeightedCoassociationModel:

	def __init__( self, k, num_documents, embedding, top_terms=10):
		self.top_terms = top_terms
		self.co_matrix = TermCoassociationMatrix()
		self.dt_matrix = DocumentTopicMatrix( num_documents )
		self.k = k
		self.term_clusters, self.partition, self.W = None, None, None
		self.dfc, self.dfs, self.L, self.dfz = None, None, None, None
		self.embedding = embedding
		self.cluster_pair_coherences, self.coherences = [], []

	def build_embedding_similarity_matrix( self, terms ):
		# create a dictionary to map terms to indices, to improve performance
		term_map = {}
		for i, term in enumerate( terms ):
			term_map[term] = i
		# populate the matrix
		S = np.zeros( ( len(terms),len(terms) ) )
		for term1, term2 in itertools.combinations(terms, 2):
			i, j = term_map[term1], term_map[term2]
			try:
				S[i,j] = max(self.embedding.similarity( term1, term2 ), 0)
				S[j,i] = S[i,j]
			except:
				# ignore missing terms
				pass
		# set the diagonal to be the maximum similarity value
		np.fill_diagonal(S, 1)    
		return S

	def add( self, rankings, partition ):
		self.co_matrix.add( rankings )
		self.dt_matrix.add( rankings, partition )

	def apply( self ):
		# Step 1. Build the term-term co-association matrix
		self.C, co_terms = self.co_matrix.build_matrix()
		self.dfc = pd.DataFrame( self.C, index=co_terms, columns=co_terms )
		log.info("Built %d x %d co-association matrix" % ( self.C.shape ) )

		# Step 2. Build the term-term similarity matrix
		self.S = self.build_embedding_similarity_matrix( co_terms  )
		self.dfs = pd.DataFrame( self.S, index=co_terms, columns=co_terms )
		log.info("Built %d x %d co-association matrix" % ( self.S.shape ) )

		# Step 3 Build the weighted co-association matrix
		self.L = np.multiply(self.C, self.S)
		self.dfl = pd.DataFrame( self.L, index=co_terms, columns=co_terms )

		# Find the term clusters first:
		log.info("Finding %d term clusters ..." % self.k)
		clusterer = TermClusterer( self.k, self.top_terms )
		self.term_clusters = clusterer.find_clusters( self.dfl )

		# Build the corresponding document representation
		Z, dt_terms = self.dt_matrix.build_matrix()
		doc_indices = range( 0, Z.shape[0] )
		self.dfz = pd.DataFrame( Z, index=doc_indices, columns=dt_terms )	
		log.info("Built %d x %d rectangular matrix" % ( Z.shape ) )

		# Now map to the documents
		num_documents = self.dt_matrix.num_documents
		# get the weights for each topic
		document_weights = []
		for i, term_cluster in enumerate(self.term_clusters):
		    topic_doc_weights = np.zeros(num_documents)
		    for term in term_cluster:
		        topic_doc_weights += self.dfz[term]
		    topic_doc_weights /= len(term_cluster)
		    document_weights.append( topic_doc_weights )	
		self.W = np.stack( document_weights, axis = 1 )	 
		self.partition = np.argmax( self.W, axis = 1 ).flatten().tolist()
		#return self.term_clusters, self.partition

		# Calculate pair coherences
		for i, term_cluster in enumerate(self.term_clusters):
			topic_dict = {}
			for term1, term2 in itertools.combinations(term_cluster, 2):
				topic_dict[(term1, term2)] = self.dfl[term1][term2]
			self.cluster_pair_coherences.append(topic_dict)

	def evaluate_term_cluster(self, df, term_cluster):
		scores = []
		for term1, term2 in itertools.combinations(term_cluster, 2):
			scores.append(df[term1][term2])
		return np.mean(scores)


