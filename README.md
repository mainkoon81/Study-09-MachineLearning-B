# Study-09-MachineLearning-[Supervised Learning]

----------------------------------------------------------------------------------------------------------------------------------------
## Linear Regression (for numeric data)
##### Single value
```
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(df[ ['predictor'] ], df[ ['response'] ])

model.predict([ [127], [248] ])
``` 
array([[ 438.94308857, 127.14839521]])
 - (The reason for predicting on an array like [127] and not just 127, is because we can have a model that makes a prediction using multiple features.) 

##### Multiple values
 - (The dataset consists of 13 features of 506 houses and their median value in $1000's. We fit a model on the 13 features to predict on the value of houses, i.e 'x' has 506 lists.) 
```
from sklearn.datasets import load_boston
boston_data = load_boston()
x = boston_data['data']
y = boston_data['target']

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x, y)

sample_house = [[2.29690000e-01, 0.00000000e+00, 1.05900000e+01, 0.00000000e+00, 4.89000000e-01,
                6.32600000e+00, 5.25000000e+01, 4.35490000e+00, 4.00000000e+00, 2.77000000e+02,
                1.86000000e+01, 3.94870000e+02, 1.09700000e+01]]

model.predict(sample_house)
```
array([ 23.68420569])

#### Regularization
 - Goal: Improve our model & Avoid overfitting.
 - Idea: Take the complexity of the model into account when we calculate the error. If we find a way to increment the error by some function of the `coefficients`, it would be great because in some way the complexity of our model will be added into the error so a complex model will have a larger error than a simple model. We respect that the simpler model have a tendency to generalize better.  
 - Question is...What if we punish the complicated model too little or punish it too much ? 
   - a model to send the rocket to the moon or a medical model have very little room for error so we're ok with some complexity.
   - a model to recommend potential friends have more room for experimenting and need to be simpler and faster to run on big data.
   - For every case, we have to tune how much we want to punish complexity in each model. This can be fixed with a parameter called 'lambda'. 
   






