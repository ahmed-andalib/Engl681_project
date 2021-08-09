import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
warnings.filterwarnings('ignore', category=DeprecationWarning)
import numpy as np
from gensim.models import Word2Vec,FastText,KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
import gensim.downloader as api
from tqdm import tqdm
from os import listdir

class Vectorizer:
    
    def __init__(self,type,pre_trained=False,retrain=False,extend_training=False,params={}):
        self.type = type
        self.pre_trained = pre_trained
        self.params = params
        self.retrain = retrain
        self.extend_training = extend_training
        self.vectorizer = None
        self.max_len = None

    #### creates the word2vec model
    #### uses pretrained embeddings
    def word2vec(self):
        if not self.pre_trained:
            if 'word2vec.model' not in listdir('./embeddings') or self.retrain:
                print('\nTraining Word2Vec model...')
                model = self.train_w2v()
            elif self.extend_training and 'word2vec.model' in listdir('./embeddings'):
                print('\nExtending existing Word2Vec model...')
                model = Word2Vec.load("./embeddings/word2vec.model")
                model.train(self.data, total_examples=len(self.data), epochs=5000)
                model.save("./embeddings/word2vec.model")
            else:
                print('\nLoading existing Word2Vec model...')
                model = Word2Vec.load("./embeddings/word2vec.model")
        else:
            model = Word2Vec(self.data,**self.params)
        vectorizer = model.wv
        self.vocab_length = len(model.wv.vocab)
        vectors = [
            np.array([vectorizer[word] for word in tweet  if word in model]).flatten() for tweet in tqdm(self.data,'Vectorizing')
            ]
        if not self.max_len:
            self.max_len = np.max([len(vector) for vector in vectors])
        self.vectors = [
            np.array(vector.tolist()+[0 for _ in range(self.max_len-len(vector))]) for vector in tqdm(vectors,'Finalizing')
            ]
        return self.vectors

    ### this method will train the word2vec model and save the model under embeddings folder, uses skip when the argument grams sg=1, CBOW when sg=0
    def train_w2v(self):
        import logging
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        
        # model = Word2Vec(self.data, sg=1,window=3,size=100,min_count=1,workers=4,iter=1000,sample=0.01)   ### skipgram
        model = Word2Vec(self.data, sg=0,window=3,size=100,min_count=1,workers=4,iter=1000,sample=0.01)     ### CBOW
        
        if self.extend_training:
            model.train(self.data, total_examples=len(self.data), epochs=500)
        model.save("./embeddings/word2vec.model")
        print("Done training w2v model!")
        return model

    ### method generates teh tfidf feature model
    def tfidf(self):
        vectorizer = TfidfVectorizer(**self.params)
        untokenized_data =[' '.join(tweet) for tweet in self.data]
        if not self.vectorizer:
            self.vectorizer = vectorizer.fit(untokenized_data)
        self.vectors = self.vectorizer.transform(untokenized_data).toarray()
        return self.vectors
    
    
    ### method create the BoW model, in the main file, we append the test BoW test vectros to match the size of training vectors
    def BoW(self):
        vectorizer = CountVectorizer(**self.params)
        untokenized_data =[' '.join(tweet) for tweet in self.data]
        if not self.vectorizer:
            self.vectorizer = vectorizer.fit(untokenized_data) 
        counts = np.array(vectorizer.transform(untokenized_data).toarray()).sum(axis=0)
        mapper = vectorizer.vocabulary_
        vectors = [
            np.array([counts[mapper[word]] for word in tweet  if word in mapper.keys()]).flatten() for tweet in tqdm(self.data,'Vectorizing')
            ]
        if not self.max_len:
            self.max_len = np.max([len(vector) for vector in vectors])
        self.vectors = [
            np.array(vector.tolist()+[0 for _ in range(self.max_len-len(vector))]) for vector in tqdm(vectors,'Finalizing')
            ]
        self.vocab_length = len(mapper.keys())
        self.words_freq = sorted([[word,counts[mapper[word]]] for word in list(mapper.keys())],key= lambda x:x[1],reverse=True)
        return self.vectors
    
    
    ### this method converts a collection of text documents to a matrix of token counts as feature vectors
    ### pasing {"ngram_range=(1, 3)","analyzer='char'"} as paramter list, we can create word or character n-grams
    def count(self):
        vectorizer = CountVectorizer(**self.params)
        untokenized_data =[' '.join(tweet) for tweet in self.data]
        if not self.vectorizer:
            self.vectorizer = vectorizer.fit(untokenized_data)
        self.vectors = self.vectorizer.transform(untokenized_data).toarray()
        self.vocab_length = len(self.vectorizer.vocabulary_.keys())
        return self.vectors
    

    ### method creates GloVe vectors from inout using the GloVe embedding from the API under gensim library
    ## it uses pretrained GloVe embeddings
    def glove(self):
        from os import listdir
        if 'glove-twitter-100.gz' in listdir('./embeddings'):
            print('\nLoading Glove Embeddings from file...')
            model = KeyedVectors.load_word2vec_format('./embeddings/glove-twitter-100.gz')
        else:
            print('\nLoading Glove Embeddings from api...')
            model = api.load('glove-twitter-100')
        vectorizer = model.wv
        vectors = [np.array([vectorizer[word] for word in tweet if word in model]).flatten() for tweet in tqdm(self.data,'Vectorizing')]
        self.vocab_length = len(model.wv.vocab)
        if not self.max_len:
            self.max_len = np.max([len(vector) for vector in vectors])
        self.vectors = [
            np.array(vector.tolist()+[0 for _ in range(self.max_len-len(vector))]) for vector in tqdm(vectors,'Finalizing')
            ]
        for i,vec in enumerate(self.vectors):
            self.vectors[i] = vec[:self.max_len]
        return self.vectors

    ### method creates FastText vectores from inout using the GloVe embedding from the API under gensim library
    ## it uses pretrained FastText embeddings
    def fasttext(self):
        if not self.pre_trained:
            if 'fasttext.model' not in listdir('./embeddings') or self.retrain:
                print('\nTraining FastText model...')
                model = self.train_ft()
            elif self.extend_training and 'fasttext.model' in listdir('./embeddings'):
                print('\nExtending existing FastText model...')
                model = FastText.load("./embeddings/fasttext.model")
                model.train(self.data, total_examples=len(self.data), epochs=100)
                model.save("./embeddings/fasttext.model")
            else:
                print('\nLoading existing FastText model...')
                model = Word2Vec.load("./embeddings/fasttext.model")
        else:
            model = FastText(self.data,**self.params)
        vectorizer = model.wv
        self.vocab_length = len(model.wv.vocab)
        vectors = [
            np.array([vectorizer[word] for word in tweet if word in model]).flatten() for tweet in tqdm(self.data,'Vectorizing')
            ]
        if not self.max_len:
            self.max_len = np.max([len(vector) for vector in vectors])
        self.vectors = [
            np.array(vector.tolist()+[0 for _ in range(self.max_len-len(vector))]) for vector in tqdm(vectors,'Finalizing')
            ]
        return self.vectors

    ### this method trains the FastText model
    def train_ft(self):
        import logging
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        model = FastText(self.data, sg=1,window=3,size=50,min_count=1,workers=4,iter=100,sample=0.01)
        if self.extend_training:
            model.train(self.data, total_examples=len(self.data), epochs=100)
        model.save("./embeddings/fasttext.model")
        print("Done training fasttext model!")
        return model

    ### wrapper method
    ### method is called with class instance from main with proper vectorizer method already specified via the constructor
    def vectorize(self,data):
        self.data = data
        vectorize_call = getattr(self, self.type, None)
        if vectorize_call:
            vectorize_call()
        else:
            raise Exception(str(self.type),'is not an available function')
        return self.vectors
    
    ### fits the model with input data
    def fit(self,data):
        self.data = data