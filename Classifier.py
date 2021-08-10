import warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore",category=FutureWarning)
warnings.filterwarnings("ignore",category=Warning)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics as mt
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np

class Classifier:
    
    def __init__(self,type,params={}):
        __classifers__ = {
        'KNN': KNeighborsClassifier,
        'M-NaiveBayes': MultinomialNB,
        'G-NaiveBayes':GaussianNB,
        'SVC': SVC,
        'DecisionTree': DecisionTreeClassifier,
        'RandomForest': RandomForestClassifier,
        'LogisticRegression': LogisticRegression,
        'MLP': MLPClassifier,
        'AdaBoost': AdaBoostClassifier,
        'Bagging': BaggingClassifier
        }
        if type not in __classifers__.keys():
            raise Exception('Available Classifiers: ',__classifers__.keys())
        self.classifier = __classifers__[type]
        self.params = params
        self.model = self.classifier(**self.params)   

    ### this is a standalone mathod to train a model with training data
    ## method returns the trained model
    def fit(self,tr_data,tr_labels):
        return self.model.fit(tr_data,tr_labels)

    ### this is a standalone method to test a model with test data
    ## method returns the mode predictions
    def predict(self,tst_data):
        return self.model.predict(tst_data)

    ### this method return the mean accuracy of the model
    def score(self,tst_data,tst_labels):
        return self.model.score(tst_data,tst_labels)

    ### this method performs GridSearchCV during model training, tuning the model hyperparameters that maximizes F1 score
    ### From Python documentation:
        # "GridSearchCV is an Exhaustive search over specified parameter values for an estimator. It implements a "fit" and a "score" method
        # The parameters of the estimator used to apply these methods are optimized by cross-validated grid-search over a parameter grid".
    ### this method is called from the main with best_only set to True. Method resturns the optimized hyper parameters specified in the tune_params over which
    ### it will perform the search. The model is trained on the optimized hyper parameter set.
    def tune(self,tr_data,tr_labels,tune_params=None,best_only=False,scoring='f1'):
        if not tune_params:
            tune_params = self.params
        tuner = GridSearchCV(self.model,tune_params,n_jobs=-1,verbose=1,scoring=scoring)
        
        ### uncomment this section for Naive Bayes
        ##//-->> shift data to 0-1 range, naive bayes can't handle negative vector elements, some vectorizer models create negative elements
        # scaler = MinMaxScaler()
        # scaler.fit(tr_data)
        # tr_data = scaler.transform(tr_data)
        # print(s_data)
            
        
        tuner.fit(tr_data,tr_labels)
        self.model = tuner.best_estimator_
        if best_only:
            return {'score':tuner.best_score_,'parmas':tuner.best_params_}
        else:
            param_scores = {}
            results = tuner.cv_results_
            for i,param in enumerate(tuner.cv_results_['params']):
                param_str  = ', '.join("{!s}={!r}".format(key,val) for (key,val) in param.items())
                param_scores[param_str]={'test_score':results['mean_test_score'][i],'train_score':results['mean_train_score'][i]}
            return param_scores
    
    ### Accessor method to get the model information
    def get_model(self):
        if getattr(self,'model',None):
            return self.model
        else:
            raise Exception('Model has not been created yet.')

    ### this method tests the model performance on test data and plots teh confusion matrix
        # tst_data contains teh test data smaples
        # tst_labels contains the true labels of test data
    ### the model calls it's appropriate predcit function and returns the predicted labels for test data
        # then the true labels and predicted labels are used to calculate and plot the confusion matrix
        # method also calculates the mean accuracy from test smaple using test data and true labels
        # method also calculates the F1 score from true and prdicted labels
    def test_and_plot(self,tst_data,tst_labels,class_num=2):
        tst_data = np.array(tst_data)
        tst_labels = np.array(tst_labels).reshape(-1,1)
        predicted_tst_labels = self.model.predict(tst_data)
        
        
        ### call to the cofusion matrix creation method
        conf_mat = mt.confusion_matrix(tst_labels, predicted_tst_labels)
        mean_accuracy = self.model.score(tst_data,tst_labels)
        f1_score = mt.f1_score(tst_labels, predicted_tst_labels)
        
        
        ### plot the confusion matrix
        self._confusion_matrix(conf_mat, labels=[i for i in range(class_num)])
        
        return conf_mat, mean_accuracy, f1_score    ### return the confusion matrix, mean accuracy and F1 score
        
       
    ### method plots the confusion matrix
    def _confusion_matrix(self,cm, title='Confusion matrix', cmap=plt.cm.Greens, labels=[]):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels, rotation=45)
        plt.yticks(tick_marks, labels)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()        
        
