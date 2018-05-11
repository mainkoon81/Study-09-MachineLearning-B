# Study-09-MachineLearning-B

----------------------------------------------------------------------------------------------------------------------------------------
## Linear Regression (for numeric data)
##### Single Predictor
```
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(df[ ['predictor'] ], df[ ['response'] ])

print(model.predict([ [127], [248] ]))
``` 
[[ 438.94308857, 127.14839521]]...The reason for predicting on an array like [127] and not just 127, is because we can have a model that makes a prediction using multiple features. 

##### Multiple Predictors
 - : The dataset consists of 13 features of 506 houses and their median value in $1000's. We fit a model on the 13 features to predict on the value of houses.
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

prediction = model.predict(sample_house)
```
array([ 23.68420569])









