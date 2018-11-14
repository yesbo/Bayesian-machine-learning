# Bayesian version of machine learning models 

## 1. Beyasian Naive Bayesian Classfier.
###Model setting: <br>
####1.Each d-dimensional vector x has a label y with y = 0,1,2,....,k
####2.Set the probability of ![alt text](http://latex.codecogs.com/gif.latex?P%28x_n%7C%5Clambda_%7By_n%7D%29%3D%5Cprod_%7Bj%3DI%7D%5EdPoisson%28x_n_%2C_j%7C%5Clambda_%7By_n_%2Cj%7D%29). <br>
####3.Let ![alt text](http://latex.codecogs.com/gif.latex?%5Cpi) be a k dimensional vector where the ith entry means P(y=i). ![alt text] (http://latex.codecogs.com/gif.latex?%5Cpi) has prior ![alt text](http://latex.codecogs.com/gif.latex?Dirichlet%28%5Calpha_1%2C...%2C%5Calpha_k%29)
####4. ![alt text](http://latex.codecogs.com/gif.latex?for%5C%2C%20i%5Cin%5C%7B1%2C...%2Ck%20%5C%7D%5C%2C%20and%20%5C%2C%20j%20%5Cin%5C%7B1%2C...%2Cd%5C%7D%2C%5C%2C%20%5Clambda_i%2C_j%20%5C%2C%20in%20%5C%2CGamma%28a%2Cb%29)

