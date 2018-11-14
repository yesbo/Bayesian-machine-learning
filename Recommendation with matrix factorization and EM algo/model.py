import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import os
from scipy.stats import norm
from scipy.stats import multivariate_normal as m_norm
import time

class EM(object):
    # EM algorithm on recommendation system with probit distribtion. Assumptions is the following
        # U is feature matrix of customers. dimension N*d
        # V is feature matrix of movies. dimension M*d.
        # u_i , v_i from norm(0, cI)
        # p(Y_ij=1)=norm.cdf(u_i dot v_i, sigma)
        # data is a pandas dataframe with 3 columns: customer_id, movie_id, is_lik.
#     def __init__(self, R, U, V, d, c, sigma):
    def __init__ (self, data, c=1, sigma=1, d=5):
        self.c=c
        self.sigma=sigma
        self.d=d
        self.N= len( data[0].unique())
        self.M=len(data[1].unique())
        self.U=np.zeros([self.N, self.d])
        self.V=np.zeros([self.M, self.d])
        for i in range(self.N):
            self.U[i]=np.random.normal(0,0.01,5)
        for j in range(self.M):
            self.V[j]=np.random.normal(0,0.01,5)
        self.R=data.pivot_table(index=[0],columns=[1],values=[2])
        self.R=self.R.replace('NaN',0).values
        self.p_list=[]
        self.customer_ids=np.sort(data[0].unique())
        self.movie_ids=np.sort(data[1].unique())
    def E_step(self):
        U=self.U
        V=self.V
        sigma=self.sigma
        R=self.R
        U_times_V_transpose=np.matmul(U, V.T)
        densities=norm.pdf(-U_times_V_transpose/sigma ) 
        cumulatives=norm.cdf(-U_times_V_transpose/sigma )
        positive_formular=U_times_V_transpose+sigma*densities/(1-cumulatives)
        positive_formular[R!=1]=0
        negative_formular=U_times_V_transpose-sigma*densities/cumulatives
        negative_formular[R!=-1]=0
        Expectations=positive_formular+negative_formular
        return Expectations
    
    def update(self, update_U=True):
        U=self.U
        V=self.V
        c=self.c
        sigma=self.sigma
        d=self.d
        N=self.N
        M=self.M
        R=self.R
        Expectations=self.Expectations
        if update_U:
            I_over_c=np.repeat( (np.identity(d)/c).reshape([1,d,d]), N, axis=0 ) # shape [N, d, d ]
            self.I_over_c=I_over_c
            Expect_times_V=np.matmul(  Expectations, V).reshape([N,1,d])
            self.Expect_times_V=Expect_times_V
            all_V_Vt=np.array( [np.outer(V[j], V[j])  for j in range(M)] ).reshape([1,M,d,d])
            self.all_V_Vt=all_V_Vt
            all_V_Vt_filter=np.abs(R).reshape( [R.shape[0], R.shape[1], 1, 1 ] ) # shape [N, M, 1, 1]
            self.all_V_Vt_filter=all_V_Vt_filter
            filtered_V_Vt=all_V_Vt*all_V_Vt_filter   # shape [N, M, d, d]
            self.filtered_V_Vt=filtered_V_Vt
            summed_V_Vt=np.sum( filtered_V_Vt, axis=1) # shape [N, d, d]
            self.summed_V_Vt=summed_V_Vt
            to_be_inverse=I_over_c+summed_V_Vt/(sigma**2) # shape [N, d, d]
            self.to_be_inverse=to_be_inverse
            inversed=np.linalg.inv(to_be_inverse)            
            self.inversed=inversed
            U=np.matmul(Expect_times_V/(sigma**2), inversed).squeeze()
            return U

        else:
            I_over_c=np.repeat( (np.identity(d)/c).reshape([1,d,d]), M, axis=0 ) # shape [M, d, d ]
            Expectations_T_time_U=np.matmul(Expectations.T, U).reshape([M,1,d]) 
            all_U_Ut=np.array( [np.outer(U[i], U[i])  for i in range(N)] ).reshape([1,N,d,d])
            all_U_Ut_filter=np.abs(R).T.reshape( [R.shape[1], R.shape[0], 1, 1 ] ) # shape [M, N, 1, 1]
            filtered_U_Ut=all_U_Ut*all_U_Ut_filter   # shape [M, N, d, d]
            summed_U_Ut=np.sum( filtered_U_Ut, axis=1) # shape [M, d, d]
            to_be_inverse=I_over_c+summed_U_Ut/sigma**2 # shape [M, d, d]
            inversed=np.linalg.inv(to_be_inverse)
            V=np.matmul(Expectations_T_time_U/sigma**2, inversed).squeeze()
            return V
        
    # calcualte In(P(R,U,V))
    def p_calculate(self):
        U=self.U
        V=self.V
        c=self.c
        sigma=self.sigma
        d=self.d
        N=self.N
        M=self.M
        R=self.R
        log_P_U=m_norm.logpdf(U,mean=np.zeros([d]),cov=np.identity(d)/c)
        log_P_U_sum=np.sum(log_P_U)
        log_P_V=P_U=m_norm.logpdf(V,mean=np.zeros([d]),cov=np.identity(d)/c)
        log_P_V_sum=np.sum(log_P_V)
        U_time_V_over_sigma=np.matmul(U,V.T)/sigma
        cumulative=norm.cdf(U_time_V_over_sigma)
        positive_filter=R==1
        negative_filter=R==-1
        P_R=cumulative*positive_filter+(1-cumulative)*negative_filter
        log_P_R=np.log(P_R)
        log_P_R[log_P_R==-np.inf]=0
        p=np.sum(log_P_R)+log_P_U_sum+log_P_V_sum
        return p
    def train(self, max_iteration=100):
        # order of iterate is: E_step -> update_U->E_step->update_V
        start=time.time()
        for itera in range(max_iteration):
            self.Expectations=self.E_step()
            update_U=True
            self.U=self.update(update_U)
            self.Expectations=self.E_step()
            update_U=False
            self.V=self.update(update_U)
            p=self.p_calculate()
            print(p)
            self.p_list.append(p)
            print( 'iteration: %s take time: %s ' %(itera, time.time()-start)  )
    def predict (self, data):
        predicts=[]
        for row in data.values:
            i=np.where(self.customer_ids==row[0])
            j=np.where(self.movie_ids==row[1])
            u_i=self.U[i]
            v_j=self.V[j]
            p=norm.cdf(np.sum(u_i*v_j)/self.sigma)
            predict=1 if p>0.5 else -1
            predicts.append(predict)
        return predicts
        
#     def predict(data):
