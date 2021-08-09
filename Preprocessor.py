import numpy as np
import copy
import re
import contractions as ct
from tqdm import tqdm
import imp
import warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning)
class Preprocessor:

    def __init__(self,*args):
        self.params =[]
        if args:
            if isinstance(args[0],tuple):
                self.params = list(*args)
            else:
                self.params = list(args)
        # self.params = ['tokenize']+self.params  ### constructor calls tokenize method


    ### implemented method: lowrcase each tweet
    def lowercase(self):
        for i,tweet in tqdm(enumerate(self.data),'LowerCase'):
            # print(tweet.lower())
            self.data[i] = tweet.lower()     ### converts tweet into lowercase and tokenize
        return self.data
    
    ### implemented method: this method used the following regular expression to clean various non-informative data from tweets
    def RegExpCleaner(self):
        # # regular expressions:
        reg_p1 = r'\b\s!'                       # remove trailing whitepsace before ! 
        reg_p2 = r'^\s*|@user|\s@user|\surl'    # remove leading whitespace, extra white spaces within tweet, @USER tag, the word url
        reg_p3= r'\.\.+'                        # remove multiple dots between words
        reg_p4 = r'\s\s+'                       # remove double or more consecutive whitespaces
        reg_p5 = r'[^a-zA-Z.,!?/:;\"\'\s]'      # remove non-alpha, numbers and non regular characters, will remove emoticons 
        reg_p6 = r'http\S+'                     # remove all urls
        
        for i,tweet in tqdm(enumerate(self.data),'Regular Expression Cleaner'):
            
            tweet = re.sub(reg_p1, '!', tweet).strip()  # remove trailing whitepsace before ! 
            tweet = re.sub(reg_p2, '', tweet).strip()   # remove leading whitespace and @USER tag
            tweet = re.sub(reg_p3, ' ', tweet).strip()  # remove multiple dots between words, replace with single whitespace
            tweet = re.sub(reg_p4, ' ', tweet).strip()  # remove double or more consecutive whitespaces , replace with single whitespace
            tweet = re.sub(reg_p5, '', tweet).strip()   # remove non-alpha and non regular characters 
            tweet = re.sub(reg_p6, '', tweet).strip()   # remove all urls
            
            self.data[i] = tweet     ### converts tweet into lowercase and tokenize
        return self.data
        
    ### implemented method: this method expands contractions   
    def ExpandContraction(self):
        for i,tweet in tqdm(enumerate(self.data),'Expand Contractions'):
  
            # print('>>> ' + tweet)
            expanded_words = []
            for word in tweet.split():
                # using contractions.fix to expand the shortened words
                expanded_words.append(ct.fix(word))   
                tweet = ' '.join(expanded_words)
            self.data[i] = tweet
            
            # print('++++ ' + tweet)
            
        return self.data
    
    # implemented method: this method removes punctuation marks 
    def RemovePuncMarks(self):
        from nltk import RegexpTokenizer
        punc_tokenizer = RegexpTokenizer(r'\w+')   # using regular expression to remove punctuations
        for i,tweet in tqdm(enumerate(self.data),'Remove Punctuations'):
            tweet = punc_tokenizer.tokenize(tweet)
            tweet = ' '.join(tweet)
            
            self.data[i] = tweet
            
        return self.data

    ### this methods tokenizes the inout tweets
    def tokenize(self):
        from nltk import word_tokenize
        for i,tweet in tqdm(enumerate(self.data),'Tokenization'):
            self.data[i] = word_tokenize(tweet)     ### converts tweet into lowercase and tokenize
        return self.data

    ### this method removes stopwards using nltk
    def remove_stopwords(self):
        from nltk.corpus import stopwords
        import re
        stop = set(stopwords.words("english"))
        noise = ['user']
        for i,tweet in tqdm(enumerate(self.data),'Stopwords Removal'):
            # print(tweet)
            self.data[i] = [w for w in tweet if w not in stop and not re.match(r"[^a-zA-Z\d\s]+", w) and w not in noise]
        return self.data
    
    
    ### refactored mathod: this method extracts POS tag using Wordnet
    def get_pos(self, word):
        from nltk import pos_tag
        from nltk.corpus import wordnet
        tag = pos_tag([word])[0][1]
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    ### refactored method: this method uses nltk WordLemmatizer to lemmatize token words
    def lemmatize(self):
        from nltk.stem import WordNetLemmatizer
        wnl = WordNetLemmatizer()
        for i, tweet in tqdm(enumerate(self.data),'Lemmatization'):
            for j, word in enumerate(tweet):
                self.data[i][j] = wnl.lemmatize(word, pos=self.get_pos(word))
        return self.data
    
     ### refactored method: applied PorterStemmer to stem words
    def stem(self):
        from nltk.stem import PorterStemmer
        stemmer = PorterStemmer()
        for i,tweet in tqdm(enumerate(self.data),'Stemming'):
            for j,word in enumerate(tweet):
                self.data[i][j] = stemmer.stem(word)
        return self.data
    
    ### refactored method: creates visual word clouds
    def word_cloud(self,labels=None,filter=None):
        if not isinstance(self.data[0],list):
            raise Exception('Data must be tokenized before using word cloud.')
        import matplotlib.pyplot as plt
        from wordcloud import WordCloud
        filters = ['NOT','UNT','TIN','GRP','OTH','OFF']
        if not filter:
            plot_data = [w for i,tweet in enumerate(self.data) for w in tweet]
        else:
            if not labels:
                raise Exception('Labels must be provided for filtering text.')
            filter = filters.index(filter)
            if filter == 4:
                plot_data = [w for i,tweet in enumerate(self.data) for w in tweet if labels[i] >0]
            else:
                plot_data = [w for i,tweet in enumerate(self.data) for w in tweet if labels[i]==filter]
        all_words = ' '.join(plot_data)
        wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)
        plt.figure(figsize=(10, 7))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis('off')
        plt.show()

    ### wrapper method, when called by main, applies the specified aequence of preprocessing steps
    def clean(self, data):
        self.data = copy.deepcopy(data)
        for param in tqdm(self.params,'Preprocessing'):
            clean_call = getattr(self, param,None)
            if clean_call:
                clean_call()    ### the core function that calls all the individual preprocessing functions
            else:
                raise Exception(str(param)+' is not an available function')
        return self.data