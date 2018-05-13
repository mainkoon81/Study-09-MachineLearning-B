# Study-09-MachineLearning-[Supervised Learning]

----------------------------------------------------------------------------------------------------------------------------------------
## Linear Regression (when 'y' follows Normal-Dist)
 - For numeric data
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

### Regularization
 - Goal: Improve our model & Avoid overfitting.
 - Idea: Take the complexity of the model into account when we calculate the error. If we find a way to increment the error by some function of the `coefficients`, it would be great because in some way the complexity of our model will be added into the error so a complex model will have a larger error than a simple model. We respect that the simpler model have a tendency to generalize better.  
 - Question is...What if we punish the complicated model too little or punish it too much ? 
   - a model to send the rocket to the moon or a medical model have very **little room for error** so we're ok with some complexity.
   - a model to recommend potential friends have **more room for experimenting** and need to be simpler and faster to run on big data.
   - For every case, we have to tune how much we want to punish complexity in each model. This can be fixed with a parameter called 'lambda'. 
     - If having a small lambda: multiply the complexity error by a small lambda (it won't swing the balance - "complex model wins".)
     - If having a large lambda: multiply the complexity error by a large lambda (it punishes the complex model more - "simple model wins".)
<img src="https://user-images.githubusercontent.com/31917400/39946131-e7c5a1da-5564-11e8-83f5-3f2e8e7c021d.jpg" />

## Linear Regression - Generalized_01 (Logistic: when 'y' follows Binomial Dist)
 - For categoric data
**[Find a DecisionSurface!]** 
> PREDICTION: based on the line best cut the data, we can guess 'pass/fail' of new student.
 - The number of errors is not what we want to minimize.
 - Instead we want to minimize sth that captures the number of errors called 'Log-loss function'.
   - The 'error function' will assign a large/small **penalty** to the incorrectly/correctly classified points.  
   - then we juggle the line around to minimize the sum of penalities(minimizing the error function)
   - Here, 'p' is the probability or proportion.
<img src="https://user-images.githubusercontent.com/31917400/39021406-93efa878-4428-11e8-8bac-04d841fbbf16.jpg" />
<img src="https://user-images.githubusercontent.com/31917400/34471521-d497e2bc-ef43-11e7-8e70-5d232b659be0.jpg" />

**Typical Approach**
 - Fitting a logistic regression to a dataset where we would like to predict if a transaction is fraud or not.
<img src="https://user-images.githubusercontent.com/31917400/34495490-4fb25f96-efed-11e7-8fb0-5eadb50da2d0.jpg" width="160" height="50" />

As we can see, there are two columns that need to be changed to dummy variables. Use the 1 for weekday and True, and 0 otherwise.
```
df['weekday'] = pd.get_dummies(df['day'])['weekday']
df[['not_fraud','fraud']] = pd.get_dummies(df['fraud'])

df = df.drop('not_fraud', axis=1)
df.head(2)
```
<img src="https://user-images.githubusercontent.com/31917400/34495708-4c4fd206-efee-11e7-8a32-1f419d1aa80e.jpg" width="200" height="50" />

The proportion of fraudulent, weekday... transactions...?
```
print(df['fraud'].mean())
print(df['weekday'].mean())
print(df.groupby('fraud').mean()['duration'])
```
<img src="https://user-images.githubusercontent.com/31917400/34495836-e1ec77ba-efee-11e7-826c-fc707de638ce.jpg" width="120" height="50" />

Fit a logistic regression model to predict if a transaction is fraud using both day and duration. Don't forget an intercept! Instead of 'OLS', we use 'Logit'
```
df['intercept'] = 1

log_model = sm.Logit(df['fraud'], df[['intercept', 'weekday', 'duration']])
result = log_model.fit()
result.summary()
```
<img src="https://user-images.githubusercontent.com/31917400/34496037-d41f3d2e-efef-11e7-85b9-d88c9d2faa30.jpg" width="400" height="100" />

Coeff-interpret: we need to exponentiate our coefficients before interpreting them.
```
# np.exp(result.params)
np.exp(2.5465)
np.exp(-1.4637), 100/23.14
```
12.762357271496972, (0.23137858821179411, 4.32152117545376)

>On weekdays, the chance of fraud is 12.76 (e^2.5465) times more likely than on weekends...holding 'duration' constant. 

>For each min less spent on the transaction, the chance of fraud is 4.32 times more likely...holding the 'weekday' constant. 

*Note: When you find the ordinal variable with numbers...Need to convert to the categorical variable, then
```
df['columns'].astype(str).value_counts()
```

**Diagnostics**
```
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
```
 - __Confusion Matrix__
   - Recall: 'reality'(Out of all the items that are **truly positive**): TP / TP+FN
   - Precision 'argued'(Out of all the items **labeled as positive**): TP / TP+FP
<img src="https://user-images.githubusercontent.com/31917400/35222988-c9570fce-ff77-11e7-82b9-7ccd3855bd50.jpg" />

 - Next, it is useful to split your data into training and testing data to assure your model can predict well not only on the data it was fit to, but also on data that the model has never seen before. Proving the model performs well on test data assures that you have a model that will do well in the future use cases. Let's pull off X and y first. Create your test set as 10% of the data, and use a random state of 0. 
```
X = df[['intercept', 'weekday', 'duration']]
y = df['fraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=0)
```
The usual steps are:
 - Instantiate
 - Fit (on train)
 - Predict (on test)
 - Score (compare predict to test)
```
log_model = LogisticRegression()
log_model.fit(X_train, y_train)
pred = log_model.predict(X_test)

print(accuracy_score(y_test, pred))
print(recall_score(y_test, pred))
print(precision_score(y_test, pred))
confusion_matrix(y_test, pred)
```
Roc Curve: The ideal case is for this to shoot all the way to the upper left hand corner. 
```
from ggplot import *
from sklearn.metrics import roc_curve, auc

preds = log_mod.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, preds)

df = pd.DataFrame(dict(fpr=fpr, tpr=tpr))
ggplot(df, aes(x='fpr', y='tpr')) + geom_line() + geom_abline(linetype='dashed')
```

## DecisionTree
 - 










-------------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------------
## Perceptron Algorithm for "Neural-Network"
 - For categoric data (Y/N)
 - Out model has 'input features', 'weights', 'bias'
<img src="https://user-images.githubusercontent.com/31917400/39957980-d507b40e-55f3-11e8-80e1-debf35768067.jpg" />
 
### Perceptron
 - Perceptron refers a combination of nodes
 - Perceptron can be a logical operator: AND, OR, NOT, XOR. They can be represented as perceptrons.
   - Take two inputs then returns an output.
<img src="https://user-images.githubusercontent.com/31917400/39961806-b1513700-5635-11e8-9edf-f3cde879577c.jpg" />

```
import pandas as pd

test_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
correct_outputs = [False, False, False, True]
outputs = []

for test_input, correct_output in zip(test_inputs, correct_outputs):
    linear_combination = weight1 * test_input[0] + weight2 * test_input[1] + bias
    output = int(linear_combination >= 0)
    is_correct_string = 'Yes' if output == correct_output else 'No'
    outputs.append([test_input[0], test_input[1], linear_combination, output, is_correct_string])

num_wrong = len([output[4] for output in outputs if output[4] == 'No'])
output_frame = pd.DataFrame(outputs, columns=['Input 1', '  Input 2', '  Linear Combination', '  Activation Output', '  Is Correct'])
if not num_wrong:
    print('Nice!  You got it all correct.\n')
else:
    print('You got {} wrong.  Keep trying!\n'.format(num_wrong))
print(output_frame.to_string(index=False))
```
1) **AND** Perceptron: weights and bias ?
```
weight1 = 1.0
weight2 = 1.0
bias = -2.0
```
2) **OR** Perceptron: weights and bias ?
 - two ways to go from an AND perceptron to an OR perceptron.
   - increase the weights
   - decrease the magnitude of the bias
```
weight1 = 2.0
weight2 = 2.0
bias = -2.0

weight1 = 1.0
weight2 = 1.0
bias = -1.0
```
3) **NOT** Perceptron: weights and bias ? 
 - the NOT operation only cares about one input. 
   - The operation returns a '0' **if the input is 1**.  
   - The operation returns a '1' **if it's a 0**. 
   - The other inputs to the perceptron are ignored. If we ignore the first input, then...
```
weight1 = 0.0
weight2 = -2.0
bias = 1.0
```
4) **XOR** Multi-Layer Perceptron(cross-OR ?)
 - What if we cannot build the decision surface ?
   - Combine perceptions: "the output of one = the input of another one"...'Neural Network'
<img src="https://user-images.githubusercontent.com/31917400/39961747-d552235e-5634-11e8-99ce-aed8a2aae548.jpg" />

### Perceptron Trick 
 - Now that we've learned that the points that are misclassified, and want the line to move closer to them. How to modify the equation of the line, so that it comes closer to a particular point?
 - Here is the example. Need to repeat this until the point becomes well-classified (For blue point, need to repeat 10 times).
<img src="https://user-images.githubusercontent.com/31917400/39961894-2f16c9f0-5638-11e8-86c0-364cec797eb0.jpg" />

### Algorithm
<img src="https://user-images.githubusercontent.com/31917400/39962325-8ef2c062-5643-11e8-8bef-c7e7adbf472d.jpg" />

> Example
<img src="https://user-images.githubusercontent.com/31917400/39962358-1e4b224a-5644-11e8-83ff-aa02165e53da.jpg" />

Recall that the perceptron step works as follows. For a **point** with coordinates(p,q), label y, and prediction given by the equation
<img src="https://user-images.githubusercontent.com/31917400/39962371-6a23c65e-5644-11e8-9a07-14f334e0ef3e.jpg" />

<img src="https://user-images.githubusercontent.com/31917400/39962396-125a87ae-5645-11e8-9253-11d4addaf568.jpg" />

```
import numpy as np
np.random.seed(42)

def stepFunction(t):
    if t >= 0:
        return 1
    return 0

def prediction(X, W, b):
    return stepFunction((np.matmul(X,W)+b)[0])    
```
The function should receive as inputs the data **X**, the labels **y**, the weights **W** (as an array), and the bias **b**. 
Update W, b, according to the perceptron algorithm, and return W and b.
```
def perceptronStep(X, y, W, b, learn_rate = 0.01):
    for i in range(len(X)):
        y_hat = prediction(X[i],W,b)
        if y[i]-y_hat == 1:
            W[0] += X[i][0]*learn_rate
            W[1] += X[i][1]*learn_rate
            b += learn_rate
        elif y[i]-y_hat == -1:
            W[0] -= X[i][0]*learn_rate
            W[1] -= X[i][1]*learn_rate
            b -= learn_rate
    return W, b
```
This function runs the perceptron algorithm repeatedly on the dataset, and returns a few of the boundary lines obtained in the iterations for plotting purposes.
> Play with the **learning rate** and the **num_epochs**.
 - 'boundary_lines' are the solution lines that get plotted below.
 - In each epoch, we apply the perceptron step.
```
def trainPerceptronAlgorithm(X, y, learn_rate = 0.01, num_epochs = 25):
    x_min, x_max = min(X.T[0]), max(X.T[0])
    y_min, y_max = min(X.T[1]), max(X.T[1])
    W = np.array(np.random.rand(2,1))
    b = np.random.rand(1)[0] + x_max
    boundary_lines = []
    for i in range(num_epochs):
        W, b = perceptronStep(X, y, W, b, learn_rate)
        boundary_lines.append((-W[0]/W[1], -b/W[1]))
    return boundary_lines
```
<img src="https://user-images.githubusercontent.com/31917400/39966015-6f875a4a-569c-11e8-804d-1b2452f3de83.jpg" />







































































