import io, os, os.path, re
import logging as log
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# --------------------------------------------------------------

token_pattern = re.compile(r"\b\w\w+\b", re.U)

def custom_tokenizer( s, min_term_length = 2 ):
	"""
	Tokenizer to split text based on any whitespace, keeping only terms of at least a certain length which start with an alphabetic character.
	"""
	return [x.lower() for x in token_pattern.findall(s) if (len(x) >= min_term_length and x[0].isalpha() ) ]

def preprocess( docs, stopwords, min_df = 3, min_term_length = 2, ngram_range = (1,1), apply_tfidf = True, apply_norm = True, tokenizer=custom_tokenizer ):
	"""
	Preprocess a list containing text documents stored as strings.
	"""
	# Build the Vector Space Model, apply TF-IDF and normalize lines to unit length all in one call
	if apply_norm:
		norm_function = "l2"
	else:
		norm_function = None
	tfidf = TfidfVectorizer(stop_words=stopwords, lowercase=True, strip_accents="unicode", tokenizer=tokenizer, use_idf=apply_tfidf, norm=norm_function, min_df = min_df, ngram_range = ngram_range) 
	X = tfidf.fit_transform(docs)
	terms = []
	# store the vocabulary map
	v = tfidf.vocabulary_
	for i in range(len(v)):
		terms.append("")
	for term in v.keys():
		terms[ v[term] ] = term
	return (X,terms)

def load_word_set( inpath ):
	"""
	Load a file of words from a file into a Python set.
	"""
	stopwords = set()
	with open(inpath) as f:
		lines = f.readlines()
		for l in lines:
			l = l.strip()
			if len(l) > 0:
				stopwords.add(l)
	return stopwords

# --------------------------------------------------------------

def save_corpus( out_prefix, X, terms, doc_ids, classes = None ):
	"""
	Save a pre-processed scikit-learn corpus and associated metadata using Joblib.
	"""
	matrix_outpath = "%s.pkl" % out_prefix 
	joblib.dump((X,terms,doc_ids,classes), matrix_outpath ) 

def load_corpus( in_path ):
	"""
	Load a pre-processed scikit-learn corpus and associated metadata using Joblib.
	"""
	(X,terms,doc_ids,classes) = joblib.load( in_path )
	return (X, terms, doc_ids, classes)


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

# --------------------------------------------------------------

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
				fin = io.open(filepath, 'r', encoding="utf8", errors='ignore')
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
			
