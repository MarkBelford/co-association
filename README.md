# Weighted Term Co-associations

Weighted Term Co-association approach for producing more coherent topics, a ranking of the topics and visualization of the topical structure.

#### Step 1: Pre-processing

Pre-process the corpus of text:
  
	python prep-text.py -o dataset --df 20 --tfidf --norm path/to/datsest 

#### Step 2: Topic Modeling 

Apply NMF to the pre-processed corpus, for the specified value or range of number of topics:

	python topic-nmf.py dataset.pkl --init random --kmin 5 --kmax 5 -r 20 --seed 1000 --maxiters 100 -o models/dataset

To check the results:

	python display-topics.py -t 10 data/bbc/nmf_k05/*rank*
  
 #### Step 3: Weighted Term Co-assocation 
 
  	python ensemble-weighted-coassoc.py -k 5 -m wikipedia2016-w2v-cbow-d100.bin -t 10 data/bbc.pkl data/bbc/nmf_k05/*partition* data/bbc/nmf_k05/*rank* -o results/bbc
 
 #### Step 4: Evaluation with Coherence
 
 Embeddings are available to download [here](http://erdos.ucd.ie/co-association/)
 
 	python evaluate-embedding.py -b -t 10 -m wikipedia2016-w2v-cbow-d100.bin -o results/bbc-coherence.csv data/bbc/nmf_k05/*rank*

 #### Step 5: Evaluation with NMI
 
  	python evaluate-accuracy.py -o results/bbc-accuracy.csv data/bbc.pkl data/bbc/nmf_k05/*partition* 
   
 #### The full weighted term co-association generation and evaluation process can be easily run with the provided jupyter notebook.
  
 
