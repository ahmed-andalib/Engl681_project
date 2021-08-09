from DataReader import DataReader
from Preprocessor import Preprocessor
from Vectorizer import Vectorizer
from Classifier import Classifier
import numpy as np
from sklearn.svm import LinearSVC
from sklearn import metrics as mt

sub_a=['NOT','OFF']


################################### Data Extraction ##########################################

### Extract the training data: tweets and their subtask A labels
dr_tr = DataReader('./datasets/training-v1/offenseval-training-v1.tsv','A')
tr_data,tr_labels = dr_tr.get_labelled_data()
tr_data,tr_labels = dr_tr.shuffle(tr_data,tr_labels,'random')

### Extract the test data: tweets and their subtask A labels
dr_tst = DataReader('./datasets/trial-data/offenseval-trial.txt', 'A')
tst_data,tst_labels = dr_tst.get_labelled_test_data()

# print(tst_labels)

### store the training data and labels into separate lists
tr_data = tr_data[:]
tr_labels = tr_labels[:]

### store the test data and labels into separate lists
tst_data = tst_data[:]
tst_labels = tst_labels[:]

# print(tst_labels)



################################### Pre Processing ##########################################

### Call to the Preprocessor class constructor
prp = Preprocessor('lowercase','RegExpCleaner', 'ExpandContraction','RemovePuncMarks', 'tokenize', 'remove_stopwords', 'lemmatize')

### call the wrapper clean method that will perform the above preprocessing steps in the ordered sequence
tr_data_clean = prp.clean(tr_data)
tst_data_clean = prp.clean(tst_data)

# print("\n",tr_data_clean)
# print("\n", tst_data_clean)

# print(tr_labels)
##############################################################################

############################ Vectorizer: Feature Vector Creation #################################

#### the Vectorizer class takes as input the name of the vectorizer to be used
# Following are the choices for vectorizers:
    # 1. Word2Vec: 
        # to use skipgram: uncomment the line  where model = Word2Ve() call is made and the sg parameter is sg=1 (default) in the train_w2v() method under
        # the Vectorizer class and comment the following line where sg=0
        # to use CBoW, comment the line where sg=1 and uncomment the following line where sg=0 in the same method under the same class
    # 2. tfidf:
        # to use tfidf change the following line: as "vct = Vectorizer('tfidf')"
    # 3. BoW:
        # to use BoW change the following line: as "vct = Vectorizer('BoW')"
    # 4. count (word unigram):
        # to use word unigram change the following line: as "vct = Vectorizer('count')"
    # 5. Word n-gram (n > 1):
        # to use word n-gram change the following line: as "vct = Vectorizer('count',{"ngram_range=(3, 3)","analyzer='word'"})" example of word tri-gram
    # 6. Character n-gram (n > 1):
        # to character n-gram change the following line: as "vct = Vectorizer('count',{"ngram_range=(3, 3)","analyzer='char'"})" example of character tri-gram
    # 7. GloVe:
        #to use gloVe vector model change the following line: as "vct = Vectorizer('glove')"
    # 8. fasttext:
        # to ue FastText vector model change the following line: as "vct = Vectorizer('fasttext')"
        
# vct = Vectorizer('count',{"ngram_range=(3, 3)","analyzer='char'"})
vct = Vectorizer('count')
tr_vectors = vct.vectorize(tr_data_clean)

### TODO: Special steps to use BoW
### We zero pad the test vectors to match dimension of training vectors as boW cannont handle this by itself


tst_vectors = vct.vectorize(tst_data_clean)  ### comment this line and uncomment below section when using BoW 


### Bow ONLY: uncomment this for BoW to pad test vectors with zeros to match train vector size
# -----------------------------------------------------#
# vct2 = Vectorizer('BoW')
# tst_vectors = vct2.vectorize(tst_data_clean)

# for i in range(len(tst_vectors)):
    
#     tst_vectors[i] = np.pad(tst_vectors[i], (0, 7), 'constant')
    

# -----------------------------------------------------#

############################################################################## 


####################################### Naive Bayes Classifier ##########################################

### create the Naive Bayes class instance
clf = Classifier('M-NaiveBayes')

### call to the tune function to train the model with optimized paramters using GridSearchCV search
tuned_accs = clf.tune(tr_vectors,tr_labels,{'alpha':[1,5,10],'fit_prior':[True,False]},best_only=True)
### print the optimized parameters
print('NB Tuned:',tuned_accs)

### test the classifier and plot the confusion matrix
cf, mean_ac, f1 = clf.test_and_plot(tst_vectors, tst_labels)

print("\nConfusion Matrix: ", cf)
print("\nMean accuracy: ", mean_ac)
print("\nF1 Score: ", f1)

##############################################################################

############################### Logistic Regression ##########################

### create the Logistic Regression class instance
clf = Classifier('LogisticRegression')

### call to the tune function to train the model with optimized paramters using GridSearchCV search
tuned_accs = clf.tune(tr_vectors,tr_labels,{'penalty':['l2'],'solver':['sag','newton-cg','lbfgs']},best_only=True)
### print the optimized parameters
print('LR Tuned:',tuned_accs)

### test the classifier and plot the confusion matrix
cf, mean_ac, f1 = clf.test_and_plot(tst_vectors, tst_labels)

print("\nConfusion Matrix: ", cf)
print("\nMean accuracy: ", mean_ac)
print("\nF1 Score: ", f1)


#################################### Random Forest ##########################################

### create the Logistic Regression class instance
clf = Classifier('RandomForest')

### call to the tune function to train the model with optimized paramters using GridSearchCV search
tuned_accs = clf.tune(tr_vectors,tr_labels,{'n_estimators':[30,40,60,160]},best_only=True)
print('RF Tuned:',tuned_accs)


### test the classifier and plot the confusion matrix
cf, mean_ac, f1 = clf.test_and_plot(tst_vectors, tst_labels)

print("\nConfusion Matrix: ", cf)
print("\nMean accuracy: ", mean_ac)
print("\nF1 Score: ", f1)


##############################################################################

# ############################ Linear SVC ###############################

### create the Linear SVC class instance
clf = LinearSVC()

### train the model using class method default hyper parameters
model = clf.fit(tr_vectors, tr_labels)

### predict the test data labels 
pr_labels = model.predict(tst_vectors)

### calculate the F1 score, mean accuracy and confusion matrix
f1 = mt.f1_score(tst_labels, pr_labels)
mean_ac = mt.accuracy_score(tst_labels, pr_labels)
cf = mt.confusion_matrix(tst_labels, pr_labels)

print("\nConfusion Matrix: ", cf)
print("\nMean accuracy: ", mean_ac)
print("\nF1 Score: ", f1)

### plot the confusion matrix
import matplotlib.pyplot as plt

labels = [0, 1]
plt.imshow(cf, interpolation='nearest', cmap=plt.cm.Reds)
plt.title('Confusion matrix')
plt.colorbar()
tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, rotation=45)
plt.yticks(tick_marks, labels)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


##############################################################################