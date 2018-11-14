# Bayesian version of machine learning models

## 1. Beyasian Naive Bayesian Classfier.
### Model setting: <br>
1.Each d-dimensional vector x has a label y with y = 0,1,2,....,k. \
2.Set the probability of ![alt text](http://latex.codecogs.com/gif.latex?P%28x_n%7C%5Clambda_%7By_n%7D%29%3D%5Cprod_%7Bj%3DI%7D%5EdPoisson%28x_n_%2C_j%7C%5Clambda_%7By_n_%2Cj%7D%29)\
3.Let ![alt text](http://latex.codecogs.com/gif.latex?%5Cpi) be a k dimensional vector where the ith entry means P(y=i)   
4.let ![alt text](http://latex.codecogs.com/gif.latex?%5Cpi) has prior ![alt text](http://latex.codecogs.com/gif.latex?Dirichlet%28%5Calpha_1%2C...%2C%5Calpha_k%29)<br>
5. ![alt text](http://latex.codecogs.com/gif.latex?for%5C%2C%20i%5Cin%5C%7B1%2C...%2Ck%20%5C%7D%5C%2C%20and%20%5C%2C%20j%20%5Cin%5C%7B1%2C...%2Cd%5C%7D%2C%5C%2C%20%5Clambda_i%2C_j%20%5C%2C%20in%20%5C%2CGamma%28a%2Cb%29)

For example, the model can be used to justify if a email is spam. y=0 or 1 where 0 means "spam email" and 1 means" spam email".
x has 54 dimentions with each entry represent the occurrence number of a sepcific word correspond to that dimension.<br>
Since Gamma-Poisson are conjugate pairs, so the posterior of lambdas is still in gamma distribution. Besides,
Dirichlet and multinominal distribution are conjugate pair too, so the posterior of
![alt text](http://latex.codecogs.com/gif.latex?%5Cpi) is still in Dirichlet.


## 2. Recommendation system with Matrix factorization and Probit-classfier.
#### Model setting: <br>

In this project, we implement an EM algorithm for the object recommendation problem. Here, we have a data set of the form R = { ![](http://latex.codecogs.com/gif.latex?r_i_j) }
restricted to a subset of pairs (i,j) ∈ Ω,


![GitHub Logo](https://github.com/yesbo/Bayesian-machine-learning/blob/master/untitled%20folder/model%20setting.png)
Format: ![Alt Text](url)

