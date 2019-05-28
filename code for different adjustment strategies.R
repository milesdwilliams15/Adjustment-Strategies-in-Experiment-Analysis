
library(tidyverse)
library(glmnet)
library(randomForest)
library(e1071)
library(np)
library(sandwich)
library(lmtest)

# Generate an outcome Y as a quadradic function of X plus stochastic noise:
N = 100 # number of observations
ATE = 2 # True treatment effect
df = tibble(
  X = rnorm(n=N,mean=50,sd=10), # X variable
  Yc = .5*X - .01*X^2 + rnorm(n=N,sd=2), # Y, untreated
  Yt = ATE + Yc, # Value of Y if treated
  Z = rbinom(n=N,size=1,prob=.25), # Random treatment assignment (25% prob of treatment)
  Y = Z*Yt + (1-Z)*Yc # Post treatment outcome
)

# Does OLS give us the right ATE?
fit1 = lm(Y ~ Z, df)
coeftest(fit1,vcovHC(fit1,type="HC2")) # Nope!

# What if I control for X (but only additively)?
fit2 = lm(Y ~ Z + X, df)
coeftest(fit2,vcovHC(fit2,type="HC2")) # Nope!!

# What if I use Lin's (2012a) technique?
fit3 = lm(Y ~ Z + X + Z*I(X - mean(X)), df)
coeftest(fit3,vcovHC(fit3,type="HC2")) # Nope!!!

# If I control for X and X^2?
fit4 = lm(log(Y) ~ Z + X + I(X^2), df)
coeftest(fit4,vcovHC(fit4,type="HC2")) # Yes!!!!


# But what do I do if I don't know how to properly adjust for X?

# Can I use a random forest to residualize Y?
rf = randomForest(Y ~ X, df)
Ypred = predict(rf)
Yresd = df$Y - Ypred
fit5 = lm(Yresd ~ Z, df)
coeftest(fit5,vcovHC(fit5,type="HC2")) # Closer than other methods, but not there yet.


# Can I use a variable selection method?
lbd = cv.glmnet(x=with(df,cbind(X,X^2,X^3,X^4,X^5)),y=df$Y)$lambda.min
lr = glmnet(x=with(df,cbind(X,X^2,X^3,X^4,X^5)),y=df$Y,alpha=1,
            lambda = lbd)
coef(lr) # X^2 and X^3 make the cut...
fit6 = lm(Y ~ Z + I(X^2) + I(X^3), df)
coeftest(fit6,vcovHC(fit6,type="HC2")) # Wrong specification, but it worked!


# What about a support vector machine (SVM)?
sv = svm(Y ~ X, df)
Ypred = predict(sv)
Yresd = df$Y - Ypred
fit7 = lm(Yresd ~ Z, df)
coeftest(fit7,vcovHC(fit7,type="HC2"))


# A non-parametric smoother?
npfit = npreg(Y ~ X, df)
Ypred = predict(npfit)
Yresd = df$Y - Ypred
fit8 = lm(Yresd ~ Z, df)
coeftest(fit8,vcovHC(fit8,type="HC2")) # Hey! Not Bad!


# Rank based transformation?
Yrank = rank(df$Y)
fit9 = lm(Yrank ~ Z, df)
coeftest(fit9,vcovHC(fit9,type="HC2"))
