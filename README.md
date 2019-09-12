# co-association

### Summary
Implementation of ensemble weighted co-association measure

Paper available at: 

#### Step 1: Pre-processing

Pre-process the corpus of text:

    python prep-text.py --tfidf --norm -o data/sample data/sample

#### Step 2: Topic Modeling 
    
Apply NMF to the pre-processed corpus, for the specified value of range of number of topics, and number of runs to generate:
    
    python generate-nmf.py data/sample.pkl --kmin 3 --kmax 8 -r 50 -o results
    
To check the results:
    
    python display-topics.py -l results/nmf_k03/ranks*.pkl
    
#### Step 3 (Optional): Build Word2Vec Model
    
 Build an appropriate Word2vec word embedding model from the corpus of text:
    
    python prep-word2vec.py --df 10 -m sg -d 100 -b -o data/model-w2v.bin data/sample/
    
#### Step 4: Generate Weighted Co-Associations

Generate the weighted co-association matrix and ensemble topics, based on a provided word embedding model.

    python generate-coassoc.py -t 10 -m embeddings/model-w2v.bin -o wcoassoc-ranks.pkl models/ranks*.pkl

#### Step 5: Evaluate Topic Coherence

Tool for evaluating the coherence of topic models, using measures based on word embeddings.

    python evaluate-embedding.py -b -t 10 -m data/model-w2v.bin -o results-embedding-k03.csv results/nmf_k03/ranks*.pkl

#### Step 6: Rank Topics and Visualize Topic Structure

Run the included jupyter notebook to produce a ranking of topics and visualize the topical structure.

This notebook requires a word embedding file in binary format and the results generated from Step 2 as it generates the weighted co-association matrix again.
    



