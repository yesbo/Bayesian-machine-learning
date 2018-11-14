import pandas as pd
import numpy as np
import math
import os
# Define the model:
class bayesian_nb_clf(object):
    # Naive bayes classifier with beyesian perspective.
   
    def __init__(self, Dirichlet_params, Gamma_params):
        # Dirichlet_params and Gamma_params are lists.
        self.pi_s=[] # list of pi_s, proportion of each category/labels.
        self.Dirichlet_params=Dirichlet_params #list of alphas, hyperparams of prior of pi_s of mutinomial distribution
        self.alpha=Gamma_params[0] # hyperparams of prior of lambda, params of poisson generating x.
        self.beta=Gamma_params[1] # same above.
        self.posterior_alphas={} 
        self.posterior_betas={}
        self.dim_N=np.nan # dimemsion of X
        self.labels=[] #  labels of categories
        self.pred_log_probs_matrix=[]
        return
    def train(self, X, labels):
        # laebls are list of intergers.
        X=np.array(X)
        N=X.shape[0]
        labels=np.array(labels).squeeze()
        self.dim_N=X.shape[1]
        self.labels= np.sort(np.unique(labels))
        # calculate pi_s/proportion and posterior beta of labels
        for i in self.labels:
            y_i=np.sum(labels==i)
            pi=(self.Dirichlet_params[i]+y_i)/(np.sum(self.Dirichlet_params)+N)
            self.pi_s.append(pi)
            self.posterior_betas['label_'+str(i)]=y_i+self.beta
        #  calculate posterior alphas
        for i in self.labels:
            for dim in range(self.dim_N):
                self.posterior_alphas['a_'+str(i)+'_'+str(dim)]=np.sum(X[labels==i,dim])+self.alpha                            
        return    
    def predict(self, X):
        X=np.array(X)
        ## calculate log_probability of an observation
        def log_probability(x):
            x=np.array(x)
            log_probs=[]
            for i in np.sort(np.unique(self.labels)):
                log_prob=0
                for dim in range(self.dim_N):
                    alpha=self.posterior_alphas['a_'+str(i)+'_'+str(dim)]
                    beta=self.posterior_betas['label_'+str(i)]
                    x_d=x[dim]
                    alpha_log_beta=alpha*np.log(beta)
                    gamma_function_combines=np.sum( [np.log(j) for j in np.arange(alpha, alpha+x_d)] )
                    x_permu=np.sum( np.log(j) for j in np.arange(1, x_d+1) )
                    alpha_plus_xd_log_beta_plus_1=(alpha+x_d)*np.log(beta+1)
                    log_prob+=(alpha_log_beta
                                +gamma_function_combines
                                -x_permu
                                -alpha_plus_xd_log_beta_plus_1 )                                                       
                log_probs.append(log_prob)
            self.pred_log_probs_matrix.append(log_probs)
            l=log_probs.index(np.max(log_probs))
            return l
        labels=np.apply_along_axis( log_probability, 1, X)
        return labels
