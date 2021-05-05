import numpy as np
import pandas as pd
import inspect
import xgboost as xgb
import networkx as nx
import matplotlib.pyplot as plt

from random import choices 
from pgmpy.models import BayesianModel
from pgmpy.estimators import BayesianEstimator
from pgmpy.estimators import TreeSearch
from pgmpy.sampling import BayesianModelSampling

from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.isotonic import IsotonicRegression

from scipy import interpolate

class bayesian_network_estimator(BaseEstimator) :

    def __init__(self,
                 clf=MLPClassifier(random_state=0, max_iter=1000, early_stopping=True),
                 random_state=0,
                 model=None,
                 root_node=None,
                 estimator_type="chow-liu",
                 use_classifier=True
                 ) :
        
        self.clf = clf
        self.random_state = random_state
        self.model = model
        self.root_node = root_node
        self.estimator_type = estimator_type
        self.use_classifier = True
        

        # checks

    def sort_cols(self, X, Y) :
        """
        Sort columns so X and Y are similar. Returns X and Y with same column order

        :param X: Dataframe X
        :param Y: Dataframe Y

        :return X, Y: Dataframes with same column order
        """

        columns = sorted(X.columns)

        return X[columns], Y[columns]


    def split_features(self, X, method='fraction_unique', cat_cols=None, min_fraction_unique=0.05, min_unique = 10):
        """
        Removes categorical features using a given method.
            X: pd.DataFrame, dataframe to remove categorical features from.
            method: split features based on fraction of unique values, no. of unique values or by passing a 
                    list of categorical columns for 'cat_cols'
            cat_cols: List of categorical columns.
            min_fraction_unique: Minimum percentage of unique values to classify as categorical feature
        """

        if method=='fraction_unique' :
            unique_fraction = X.apply(lambda col: len(pd.unique(col))/len(col)) 
            reduced_X = X.loc[:, unique_fraction > min_fraction_unique]

        if method=='named_columns' :
            non_cat_cols = [col not in cat_cols for col in X.columns]
            reduced_X = X.loc[:, non_cat_cols]
            
        if method=='n_unique_values' :
            n_unique = df.nunique()
            reduced_X = X.loc[:, n_unique > min_unique]
            
        print(reduced_X.columns)
    
        cat_X = df[df.columns[~df.columns.isin(reduced_X)]]
    
        return reduced_X, cat_X


    def fit(self, X, random_node=False) :
        """
        Fit network

        Will later be used in combination with KDE
        """


        print(f"""Finding Bayesian Network with root node '{self.root_node}'
        Method: {self.estimator_type}
        ...""")
        self.configure_network(X,self.root_node, self.estimator_type)



        return self

    def draw_network(self, dag) :
        """
        Draw BN network
        """
        
        nx.draw(dag, with_labels=True, arrowsize=30, node_size=800, alpha=0.3, font_weight='bold')
        plt.show()

    def configure_network(self, X, root_node, estimator_type, class_node=None, draw_dag=True) :
        """
        Learn structure of data and fit a Bayesian Network model, default method is TreeSearch

        :param X: pandas DataFrame, shape (n_samples, n_features)
        :param root_node: str, int. Root node of the tree structure.
        :param estimator_type: str (chow-liu | tan). The algorithm to use for estimating the DAG.
        :param class_node: str, int. Required if estimator_type = 'tan'.
        
        :return: self : object
        """


        est = TreeSearch(X, root_node)
        dag = est.estimate(estimator_type=estimator_type, class_node=class_node)

        model = BayesianModel(dag.edges())
        model.fit(X, estimator=BayesianEstimator, prior_type='dirichlet', pseudo_counts=0.1)

        self.dag = dag
        self.model = model

        if draw_dag :
            self.draw_network(self.dag)

        return self

    def sample(self, n_samples=1) :
        """
        Sample n data points from the Bayesian Network

        :param n_samples: int, amount of datapoints to generate.
        :return: Dataframe of new datapoints shape (n_samples,n_features)
        """
        np.random.seed(self.random_state)

        inference = BayesianModelSampling(self.model)
        
        Y = inference.forward_sample(size=n_samples, return_type='dataframe')

        Y = Y[sorted(Y.columns)]
    
        return Y[cols]

    def fit_classifier(self, X0, X1) :
        """
        Fit a classifier on two classes.
        """
        
        X0, X1 = self.sort_cols(X0,X1)

        zeros = np.zeros(len(X0))
        ones = np.ones(len(X0))
    
        y = np.concatenate([zeros,ones], axis=0)
        X = np.concatenate([X0, X1], axis=0)
    
        self.clf = self.clf.fit(X,y)
        

    def get_probabilities(self, X0, X1) :
        """
        calculate probabilities of target
        """


        p0 = self.clf.predict_proba(X0)[:, 1]
        p2 = self.clf.predict_proba(X1)[:, 1]
    
        return p0, p2

    def sample_weighted(self, n, X1) :
        """
        Sample from BN and reweight
        """

        gen_sample = self.sample(n)

        self.fit_classifier(gen_sample, X1) 

        p0, p2 = self.get_probabilities(gen_sample, X1)

        p1f_ = self.isotonic_regression(p0,p2)

        weights = self.weights()

    def isotonic_regression(self, p0, p2) :
        
        nbins = 100
        hist_p0, bin_edges = np.histogram(p0, bins=nbins, range=(0, 1))
        hist_p1, bin_edges = np.histogram(p2, bins=nbins, range=(0, 1))
        bin_centers = bin_edges[:-1] + 0.5/nbins

        hnorm_p0 = hist_p0 / sum(hist_p0)
        hnorm_p1 = hist_p1 / sum(hist_p1)
        hnorm_sum = hnorm_p0 + hnorm_p1
        p1cb = np.divide(hnorm_p1, hnorm_sum, out=np.zeros_like(hnorm_p1), where=hnorm_sum != 0)

        iso_reg = IsotonicRegression().fit(bin_centers, p1cb)
        p1pred = iso_reg.predict(bin_centers)


        p1f_ = interpolate.interp1d(bin_edges[:-1], p1pred, kind='previous', bounds_error=False, 
                                         fill_value="extrapolate") 

        return p1f_
        
    def weights(self, X, clf, p1f_) :
    
        p0 = clf.predict_proba(X)[:, 1]
        nominator = p1f_(p0)
        denominator = 1 - nominator
        weight = np.divide(nominator, denominator, out=np.ones_like(nominator), where=denominator != 0)

        return weight

    def reweight_sample(self, X, weight) :
    
        pop = np.asarray(range(X.shape[0]))
        probs = weight/np.sum(weight)
        sample = choices(pop, probs, k=X.shape[0])
        Xtrans = X[sample]
        Xtrans

        return Xtrans

    def plot_hist(self, x1, x2, nbins) :
    
        plt.figure(figsize=(12,7))
        plt.hist(x1, bins=nbins, range=(0,1), alpha=0.5, log=True, density=True) 
        plt.hist(x2, bins=nbins, range=(0,1), alpha=0.5, log=True, density=True)

        plt.show()