import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE


class TreeTester():
    '''
    Object for train and test of a decision tree model.
    
    Arguments:
        features (list of strings): list of features to use as predictors (['feature0','feature1',...,]);
        train_size (float): fraction of dataset to use for train purposes;
        robustness_iteration (int): number of different train and test routines to do with a fixed configuration;
        auto_train (bool): boolean flag; allows to use the model predictions as added features;
        added_features (int): number of prediction routines to use ad added features;
        oversample (bool): activate/deactive oversampling with SMOTE technique;
        data (pandas DataFrame): original dataset
        target (string): target column.
    '''
    def __init__(self,*args,**kwargs):
        # Set default parameters
        self.features = None
        self.train_size = 0.66
        self.robustness_iterations = 100
        self.auto_train = False
        self.added_features = 0
        self.oversample = False
        self.data = pd.DataFrame()
        self.X_train = pd.DataFrame() 
        self.X_test = pd.DataFrame() 
        self.y_train = pd.DataFrame() 
        self.y_test = pd.DataFrame() 
        self.y_pred = pd.DataFrame() 
        self.target = ''
        self.classifier = tree.DecisionTreeClassifier()
        self.classifier_auto = tree.DecisionTreeClassifier()
        # Update with provided parameters
        self.__dict__.update(kwargs)
        # Actual initialization
        self.original_rows = len(self.data)
        self.X = self.data.drop([self.target],axis=1)
        self.y = self.data[self.target]
        self.evaluators = {}
        
        
    def train(self):
        '''
        Single train and test of the model.
        '''
        # oversampling, if needed
        X = self.X
        y = self.y
        if self.oversample:
            oversample = SMOTE()
            X,y = oversample.fit_resample(X,y)
        # feature selection
        if self.features:
            X_train, X_test, y_train, y_test = train_test_split(X[self.features], y, train_size=self.train_size,
                                                                test_size=1-self.train_size)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=self.train_size,
                                                                test_size=1-self.train_size)
        # using model's guess as a feature
        if self.auto_train:
            for j in range(self.added_features):
                self.classifier.fit(X_train, y_train)
                self.classifier_auto.fit(X_train, self.classifier.predict(X_train)==y_train)
                X_train[f'pred_{j}'] = self.classifier.predict(X_train)==y_train
                X_test[f'pred_{j}'] = self.classifier_auto.predict(X_test)
        self.classifier.fit(X_train, y_train)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        
    def evaluate(self):
        '''
        Constructs evaluators to rate model's performance.
        '''
        accuracy = []
        recall = []
        selectivity = []
        precision = []
        F1_measure = []
        NPV = []
        FNR = []
        for i in range(self.robustness_iterations):
            self.train()
            results = self.y_test.to_frame().rename(columns={self.target: "true_value"})
            self.y_pred = self.classifier.predict(self.X_test)
            results['predicted'] = self.y_pred
            cm = confusion_matrix(results.query('index<@self.original_rows').true_value,results.query('index<@self.original_rows').predicted)
            TP = cm[0][0]
            FP = cm[0][1]
            FN = cm[1][0]
            TN = cm[1][1]
            accuracy.append((TP+TN)/(TP+TN+FP+FN))
            recall.append(TP/(TP+FN))
            selectivity.append(TN/(TN+FP))
            precision.append(TP/(TP+FP))
            NPV.append(TN/(TN+FN))
            FNR.append(FN/(FN+TP))
            F1_measure.append(2*precision[-1]*recall[-1]/(precision[-1]+recall[-1]))
        self.evaluators['accuracy'] = accuracy
        self.evaluators['recall'] = recall
        self.evaluators['selectivity'] = selectivity
        self.evaluators['precision'] = precision
        self.evaluators['NPV'] = NPV
        self.evaluators['FNR'] = FNR
        self.evaluators['F1_measure'] = F1_measure
        self.evaluators = pd.DataFrame.from_dict(self.evaluators)
        
    
    def plot_results(self):
        '''
        Simple box-and-whiskers plot.
        '''
        fig,ax = plt.subplots()
        sns.boxplot(x="variable", y="value", data=pd.melt(self.evaluators))
        return