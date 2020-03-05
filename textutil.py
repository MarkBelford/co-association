import os, os.path, re
from sklearn.feature_extraction.text import TfidfVectorizer
import logging as log
import joblib

# --------------------------------------------------------------

token_pattern = re.compile(r"\b\w\w+\b", re.U)

def preprocess( docs, stopwords, min_df = 3, min_term_length = 2, ngram_range = (1,1), apply_tfidf = True, apply_norm = True ):
	"""
	Preprocess a list containing text documents stored as strings.
	"""	
	def custom_tokenizer( s ):
		return [x.lower() for x in token_pattern.findall(s) if (len(x) >= min_term_length and x[0].isalpha() ) ]

	# Build the Vector Space Model, apply TF-IDF and normalize lines to unit length all in one call
	if apply_norm:
		norm_function = "l2"
	else:
		norm_function = None
	tfidf = TfidfVectorizer(stop_words=stopwords, lowercase=True, strip_accents="unicode", tokenizer=custom_tokenizer, use_idf=apply_tfidf, norm=norm_function, min_df = min_df, ngram_range = ngram_range) 
	X = tfidf.fit_transform(docs)
	terms = []
	# store the vocabulary map
	v = tfidf.vocabulary_
	for i in range(len(v)):
		terms.append("")
	for term in v.keys():
		terms[ v[term] ] = term
	return (X,terms)

def preprocess_simple( docs, stopwords, min_df = 3, min_term_length = 2, ngram_range = (1,1), apply_tfidf = True, apply_norm = True ):
	"""
	Preprocess a list containing text documents stored as strings, where the documents have already been tokenized and are separated by whitespace
	"""
	token_pattern = re.compile(r"[\s\-]+", re.U)

	def custom_tokenizer( s ):
		return [x.lower() for x in token_pattern.split(s) if (len(x) >= min_term_length) ]

	# Build the Vector Space Model, apply TF-IDF and normalize lines to unit length all in one call
	if apply_norm:
		norm_function = "l2"
	else:
		norm_function = None
	tfidf = TfidfVectorizer(stop_words=stopwords, lowercase=True, strip_accents="unicode", tokenizer=custom_tokenizer, use_idf=apply_tfidf, norm=norm_function, min_df = min_df, ngram_range = ngram_range) 
	X = tfidf.fit_transform(docs)
	terms = []
	# store the vocabulary map
	v = tfidf.vocabulary_
	for i in range(len(v)):
		terms.append("")
	for term in v.keys():
		terms[ v[term] ] = term
	return (X,terms)

def preprocess_tweets( docs, stopwords, min_df = 3, min_term_length = 2, ngram_range = (1,1), apply_tfidf = True, apply_norm = True):
	"""
	Preprocess a list containing text documents stored as strings, where the documents have already been tokenized and are separated by whitespace
	"""
	from nltk.tokenize import TweetTokenizer
	tweet_tokenizer = TweetTokenizer(preserve_case = False, strip_handles=True, reduce_len=True)

	def custom_tokenizer( s ):
		# need to manually replace quotes
		s = s.replace("'"," ").replace('"',' ')
		tokens = []
		for x in tweet_tokenizer.tokenize(s):
			if len(x) >= min_term_length:
				if x[0] == "#" or x[0].isalpha():
					tokens.append( x )
		return tokens

	# Build the Vector Space Model, apply TF-IDF and normalize lines to unit length all in one call
	if apply_norm:
		norm_function = "l2"
	else:
		norm_function = None
	tfidf = TfidfVectorizer(stop_words=stopwords, lowercase=True, strip_accents="unicode", tokenizer=custom_tokenizer, use_idf=apply_tfidf, norm=norm_function, min_df = min_df, ngram_range = ngram_range) 
	X = tfidf.fit_transform(docs)
	terms = []
	# store the vocabulary map
	v = tfidf.vocabulary_
	for i in range(len(v)):
		terms.append("")
	for term in v.keys():
		terms[ v[term] ] = term
	return (X,terms)

# --------------------------------------------------------------

def load_word_list( inpath ):
	"""
	Load stopwords from a file into a set.
	"""
	stopwords = set()
	with open(inpath) as f:
		lines = f.readlines()
		for l in lines:
			l = l.strip().lower()
			if len(l) > 0:
				stopwords.add(l)
	return stopwords

def save_corpus( out_prefix, X, terms, doc_ids, classes = None ):
	"""
	Save a pre-processed scikit-learn corpus and associated metadata using Joblib.
	"""
	matrix_outpath = "%s.pkl" % out_prefix 
	log.info( "Saving document-term matrix to %s" % matrix_outpath )
	joblib.dump((X,terms,doc_ids,classes), matrix_outpath ) 

def load_corpus( in_path ):
	"""
	Load a pre-processed scikit-learn corpus and associated metadata using Joblib.
	"""
	(X,terms,doc_ids,classes) = joblib.load( in_path )
	return (X, terms, doc_ids, classes)

def find_documents( root_path ):
	"""
	Find all files in the specified directory and its subdirectories, and store them as strings in a list.
	"""
	filepaths = []
	for dir_path, subFolders, files in os.walk(root_path):
		for filename in files:
			if filename.startswith(".") or filename.startswith("_"):
				continue
			filepath = os.path.join(dir_path,filename)
			filepaths.append( filepath )
	filepaths.sort()
	return filepaths	

def read_text( in_path ):
	"""
	Read and normalize body text from the specified document file.
	"""
	http_re = re.compile('https?[:;]?/?/?\S*')
	# read the file
	f = open(in_path, 'r', encoding="utf8", errors='ignore')
	body = ""
	while True:
		line = f.readline()
		if not line:
			break
		# Remove URIs at this point (Note: this simple regex captures MOST URIs but may occasionally let others slip through)
		normalized_line = re.sub(http_re, '', line.strip())
		if len(normalized_line) > 1:
			body += normalized_line
			body += "\n"
	f.close()	
	return body
	
# --------------------------------------------------------------

def find_documents( root_path ):
	"""
	Find all files in the specified directory and its subdirectories, and store them as strings in a list.
	"""
	filepaths = []
	for dir_path, subFolders, files in os.walk(root_path):
		for filename in files:
			if filename.startswith(".") or filename.startswith("_"):
				continue
			filepath = os.path.join(dir_path,filename)
			filepaths.append( filepath )
	filepaths.sort()
	return filepaths	
	
def custom_tokenizer( s, min_term_length ):
	return [x.lower() for x in token_pattern.findall(s) if (len(x) >= min_term_length and x[0].isalpha() ) ]

class DocumentBodyGenerator:

	def __init__( self, dir_paths, min_doc_length ):
		self.dir_paths = dir_paths
		self.min_doc_length = min_doc_length

	def __iter__( self ):
		for in_path in self.dir_paths:
			# Find all text files in the directory
			log.info( "Processing %s ..." % ( in_path ) )
			for filepath in find_documents( in_path ):
				doc_id = os.path.splitext( os.path.basename( filepath ) )[0]
				fin = open(filepath, 'r', encoding="utf8", errors='ignore')
				body = fin.read()
				fin.close()
				if len(body) < self.min_doc_length:
					continue
				yield (doc_id,filepath,body)

class DocumentTokenGenerator:

	def __init__( self, dir_paths, min_doc_length, stopwords = set() ):
		self.dir_paths = dir_paths
		self.min_doc_length = min_doc_length
		self.stopwords = stopwords
		self.min_term_length = 2
		self.placeholder = "<stopword>"
		self.num_documents = 0

	def __iter__( self ):
		if len(self.dir_paths) == 1 and self.dir_paths[0].endswith('.txt'): # if only one reference file is found then assume all documents are stored in it, one per line.
			self.num_documents = 0
			with open(self.dir_paths[0]) as f:
				for line in f:
					body = line.lower().strip()
					self.num_documents += 1
					tokens = []
					for tok in custom_tokenizer( body, self.min_term_length ):
						if tok in self.stopwords:
							tokens.append( self.placeholder )
						else:
							tokens.append( tok )
					yield tokens

		else:
			bodygen = DocumentBodyGenerator( self.dir_paths, self.min_doc_length )
			self.num_documents = 0
			for doc_id, filepath, body in bodygen:
				body = body.lower().strip()
				self.num_documents += 1
				tokens = []
				for tok in custom_tokenizer( body, self.min_term_length ):
					if tok in self.stopwords:
						tokens.append( self.placeholder )
					else:
						tokens.append( tok )
				yield tokens
