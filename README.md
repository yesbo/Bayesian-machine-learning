# Bayesian-machine-learning
Bayesian version of main stream machine learning models

1. Beyasian version of Naive Bayesian Classfier.
   The model setting is as followings:
   Each d-dimensional vector x has a label y with y = 0,1,2,3.....
   For example, y=0 indicating “non-spam email” and y = 1 indicating“spam email”. We model the nth feature vector of a spam      email as
  54
p(xn|⃗λ1, yn = 1) = 􏰅 Poisson(xn,d|λ1,d), d=1
and similarly for class 0. We model the labels as yn ∼ Bernoulli(π). Assume independent gamma priors on all λ1,d and λ0,d, as in Problem 3, with a = 1 and b = 1. For the label bias assume the prior π ∼ Beta(e,f) and set e = f = 1.
Let (x∗,y∗) be a new test pair. The goal is to predict y∗ given x∗. To do this we use the predictive distribution under the posterior of the naive Bayes classifier. That is, for possible label y∗ = y ∈ {0,1} we compute
p(y∗ = y|x∗, X, ⃗y) ∝ p(x∗|y∗ = y, {xi : yi = y})p(y∗ = y|⃗y)
where X and ⃗y contain N training pairs of the form (xi,yi). This can be calculated as follows:
54 􏰆 ∞
p(x∗|y∗ = y, {xi : yi = y}) = 􏰅 p(x∗|λy,d)p(λy,d|{xi : yi = y})dλ
d=1 0
The results from Problem 3 can be directly applied here. Also, as discussed in the notes
0
∗􏰆1∗
p(y = y|⃗y) =
which has the solutions p(y∗ = 1|⃗y) = e + 􏰄n 1(yn = 1) and p(y∗ = 0|⃗y) = f + 􏰄n 1(yn = 0).
p(y = y|π)p(π|⃗y)dπ
  N+e+f N+e+f
