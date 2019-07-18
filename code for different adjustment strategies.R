
library(tidyverse)
library(glmnet)
library(randomForest)
library(e1071)
library(np)
library(sandwich)
library(lmtest)

# Generate an outcome Y as a quadradic function of X plus stochastic noise:
N = 1000 # number of observations
ATE = 5 # True treatment effect
df = tibble(
  X = rnorm(n=N,mean=50,sd=10), # X variable
  Yc = 1 + .3*(X - mean(X))^2 + rnorm(n=N,sd=10), # Y, untreated
  Yt = ATE + Yc, # Value of Y if treated
  Z = rbinom(n=N,size=1,prob=pnorm(scale(X))), # non-random treatment assignment (25% prob of treatment)
  Y = Z*Yt + (1-Z)*Yc # Post treatment outcome
)

# Does OLS give us the right ATE?
fit1 = lm(Y ~ Z, df)
coeftest(fit1,vcovHC(fit1,type="HC2")) # Nope!

# What if I control for X (but only additively)?
fit2 = lm(Y ~ Z + X, df)
coeftest(fit2,vcovHC(fit2,type="HC2")) # Nope!!

# What if I use Lin's (2012a) technique?
fit3 = lm(Y ~ Z*I(X - mean(X)), df)
coeftest(fit3,vcovHC(fit3,type="HC2")) # Nope!!!

# If I control for X and X^2?
fit4 = lm(log(Y) ~ Z + X + I(X^2), df)
coeftest(fit4,vcovHC(fit4,type="HC2")) # Nope!!!!

# Let's take a different approach

# Can I use a random forest to residualize Y?
rf = randomForest(Y ~ X, df)
Ypred = predict(rf)
Yresd = df$Y - Ypred
rf = randomForest(Z ~ X, df)
Zpred = predict(rf)
Zresd = df$Z - Zpred
fit5 = lm(Yresd ~ Zresd)
coeftest(fit5,vcovHC(fit5,type="HC2")) # Closer than other methods, but not there yet.


# What about a support vector machine (SVM)?
sv = svm(Y ~ X, df)
Ypred = predict(sv)
Yresd = df$Y - Ypred
sv = svm(Z ~ X, df)
Zpred = predict(sv)
Zresd = df$Z - Zpred
fit7 = lm(Yresd ~ Zresd, df)
coeftest(fit7,vcovHC(fit7,type="HC2"))


# A non-parametric smoother?
npfit = npreg(Y ~ X, df)
Ypred = predict(npfit)
Yresd = df$Y - Ypred
npfit = npreg(Z ~ X, df)
Zpred = predict(npfit)
Zresd = df$Z - Zpred
fit8 = lm(Yresd ~ Z, df)
coeftest(fit8,vcovHC(fit8,type="HC2")) # Hey! Not Bad!


# Rank based transformation?
Yrank = rank(df$Y)
rfit = npreg(Yrank ~ rank(X), df)
Ypred = predict(rfit)
Yresd = Yrank - Ypred
fit9 = lm(Yresd ~ Z, df)
coeftest(fit9,vcovHC(fit9,type="HC2"))
