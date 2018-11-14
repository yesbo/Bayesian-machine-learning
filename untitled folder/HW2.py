
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import os
from scipy.stats import norm
from scipy.stats import multivariate_normal as m_norm


# In[2]:


ratings=pd.read_csv('./movies_csv/ratings.csv',header=None)


# Initial parameters

# In[12]:


# initial parameters
U,V={},{}
d=5
sigma,c=1,1
customer_id=np.sort(ratings[0].unique())
movie_id=np.sort(ratings[1].unique())
for i in customer_id:
    U[i]=np.random.normal(0,0.01,5)
for j in movie_id:
    V[j]=np.random.normal(0,0.01,5)


# E-step

# In[11]:


# E-step:
def phi_calculate(index):
    i=index[0]
    j=index[1]
    u_i=U[i]
    v_j=V[j]
    ui_dot_vj=np.sum(u_i*v_j)
    selector=np.logical_and(ratings[0]==i,ratings[1]==j)
    if ratings.loc[selector,2].values>0:
        e_phi=ui_dot_vj+sigma*norm.pdf(ui_dot_vj/sigma)/ norm.cdf(ui_dot_vj/sigma)
    else:
        e_phi=ui_dot_vj-sigma*norm.pdf(ui_dot_vj/sigma)/ ( 1-norm.cdf(ui_dot_vj/sigma) )
    return e_phi
ratings['phi']=ratings.loc[:,[0,1]].apply(phi_calculate,axis=1)


# M-step

# In[5]:


# M_step:
    # Update U
def update_ui(i):
    I_over_c=np.identity(5)/c
    v_matrix=np.matrix(np.zeros([d,d]))
    j_list=ratings[1][ratings[0]==i]
    for j in j_list.values:
        v_j=V[j]
        v_matrix+=np.outer(v_j,v_j)

    inverse_matrix=np.linalg.inv(I_over_c+v_matrix/(sigma**2) )
    phi_time_vj=0
    for j in j_list.values:
        v_j=V[j]
        selector=np.logical_and(ratings[0]==i,ratings[1]==j)
        phi_i_j=phis[ ratings.index[selector].values[0] ]
        phi_time_vj+=phi_i_j*v_j
    right_part=phi_time_vj/sigma**2
    u_i=np.array(np.matmul(inverse_matrix,right_part)).squeeze()
    
    return u_i

u_index=list(U.keys())
u_i_list=list(map(update_ui,u_index))
for i in range(len(u_index)):
    index=u_index[i]
    U[index]=u_i_list[i]


# In[309]:


# M-step update v:
def update_vj(j):
    I_over_c=np.identity(5)/c
    u_matrix=np.matrix(np.zeros([d,d]))
    i_list=ratings[0][ratings[1]==j]
    for i in i_list.values:
        u_i=U[i]
        u_matrix+=np.outer(u_i,u_i)
    inverse_matrix=np.linalg.inv(I_over_c+u_matrix/(sigma**2) )
    phi_time_ui=0
    for i in i_list.values:
        u_i=U[i]
        selector=np.logical_and(ratings[1]==j,ratings[0]==i)
        phi_i_j=phis[ ratings.index[selector].values[0] ]
        phi_time_ui+=phi_i_j*u_i
    right_part=phi_time_ui/sigma**2
    v_j=np.array(np.matmul(inverse_matrix,right_part)).squeeze()
    return v_j

v_index=list(V.keys())
V_j_list=list(map(update_vj,v_index))
for i in range(len(v_index)):
    index=v_index[i]
    V[index]=V_j_list[i]


# In[14]:


# calcualte In(P(R,U,V))
from scipy.stats import multivariate_normal as m_norm
def p_calculate(row):
    i=row[0]
    j=row[1]
    r_i_j=row[2]
    u_i=U[i]
    v_j=V[j]
    ui_dot_vj=np.sum(u_i*v_j)
    p_ui=m_norm.pdf(u_i,mean=np.zeros([5]),cov=np.identity(5)/c)
    p_vj=m_norm.pdf(v_j,mean=np.zeros([5]),cov=np.identity(5)/c)
    #     print(p_ui)
    # print(u_i,p_ui)
    if r_i_j>0:
        p=norm.pdf(ui_dot_vj/sigma)*p_ui*p_vj
    else:
        p=(1-norm.pdf(ui_dot_vj/sigma))*p_ui*p_vj
    print(np.log(p))
    return np.log(p)
p=list(map(p_calculate,list(ratings.values)))
p=np.sum(p)


# In[ ]:


p_list=[]
for t in range(100):
    # phi
    phis=list(map(phi_calculate,phi_indexes))
    # update U
    u_i_list=list(map(update_ui,u_index))
    for i in range(len(u_index)):
        index=u_index[i]
        U[index]=u_i_list[i]
    # update V
    V_j_list=list(map(update_vj,v_index))
    for i in range(len(v_index)):
        index=v_index[i]
        V[index]=V_j_list[i]
    
    #calcualte p(R,V,U)
    p=list(map(p_calculate,list(ratings.values)))
    p=np.sum(p)
    print(p)
    p_list.append(p)
plt.plot(np.range(99,p_list[1:]))

