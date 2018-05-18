# Study-09-MachineLearning-[Supervised Learning]
 - Let's find the Decision Surface!
----------------------------------------------------------------------------------------------------------------------------------------
## (A1) Linear Regression (when 'y' follows Normal-Dist)
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
 - **L1 regularization** is useful for feature selection, as it tends to turn the less relevant weights into zero.
 - Question is...What if we punish the complicated model too little or punish it too much ? 
   - a model to send the rocket to the moon or a medical model have very **little room for error** so we're ok with some complexity.
   - a model to recommend potential friends have **more room for experimenting** and need to be simpler and faster to run on big data.
   - For every case, we have to tune how much we want to punish complexity in each model. This can be fixed with a parameter called 'lambda'. 
     - If having a small lambda: multiply the complexity error by a small lambda (it won't swing the balance - "complex model wins".)
     - If having a large lambda: multiply the complexity error by a large lambda (it punishes the complex model more - "simple model wins".)
<img src="https://user-images.githubusercontent.com/31917400/39946131-e7c5a1da-5564-11e8-83f5-3f2e8e7c021d.jpg" />

## (A2) Linear Regression - Generalized_01 (Logistic: when 'y' follows Binomial Dist)
 - For categoric data
 - For **binary** classification

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
--------------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## (B) DecisionTree
 - For categoric data
 - For **non-binary** classification
 
> PREDICTION: based on the features, we can guess the apps that the future users would download.  

Unlike SVM using a kernel trick, **DecisionTree** use a trick that lets a linear-DecisionSurf do Non-Linear-Decision making. 
<img src="https://user-images.githubusercontent.com/31917400/38253495-b3ae81f2-374e-11e8-8721-1a2ab32bd310.jpg" /> 
<img src="https://user-images.githubusercontent.com/31917400/39018803-08c158d6-441f-11e8-88f0-bc56f56d0df4.jpg" />  

```
import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
import sys
from class_vis import prettyPicture, output_image
from prep_terrain_data import makeTerrainData

features_train, labels_train, features_test, labels_test = makeTerrainData()
```
We build two DecisionTree classifiers; one with parameter(min_samples_split=2), and the other with (min_samples_split=50). What's the difference in accuracy ? And how to prevent **overfitting** ? 

<img src="https://user-images.githubusercontent.com/31917400/38373940-5d7c55fa-38ea-11e8-936f-7de3c3455e36.jpg" width="300" height="170" /> 

Store your predictions in a list named 'pred_2', 'pred_50'.
```
from sklearn import tree

clf_2 = tree.DecisionTreeClassifier(min_samples_split=2)
clf_50 = tree.DecisionTreeClassifier(min_samples_split=50)

X = features_train
y = labels_train

clf_2.fit(X, y)
clf_50.fit(X, y)

pred_2 = clf_2.predict(features_test)
pred_50 = clf_50.predict(features_test)
```
Accuracy ? Whose accuracy is better ? clf_2 or clf_50 ? Well..min_samples_split=2 is too much..overfitting giving less accuracy.
```
from sklearn.metrics import accuracy_score

acc_min_samples_split_2 = accuracy_score(pred_2, labels_test)
acc_min_samples_split_50 = accuracy_score(pred_50, labels_test)

def submitAccuracies():
  return {"acc_min_samples_split_2":round(acc_min_samples_split_2, 3),
          "acc_min_samples_split_50":round(acc_min_samples_split_50, 3)}
```
### DecisionTree & Entropy
 - **Entropy:** is a measure of **[impurity]** in a bunch of examples...Let's say it's an opposite of purity..   
 - **Entropy** controls how a DecisionTree decides **where to split the data** to make subsets as pure as possible...
 - **Entropy** describes what's going on here.. `- proportion_a * log(proportion_a)` `- proportion_b * log(proportion_b)`
<img src="https://user-images.githubusercontent.com/31917400/40003016-329ba1ba-578a-11e8-8fa4-ac39a8b00d21.jpg" /> 
<img src="https://user-images.githubusercontent.com/31917400/38379058-77c82d46-38f7-11e8-97f3-4583e6b0255b.jpg" />  

If we have a categorical veriable that consists of entry (a, b). Let's say p(a)=0.5, p(b)=0.5, then our entropy is
```
import math

-0.5*math.log(0.5, 2) -0.5*math.log(0.5, 2)
```
Which is 1, so it's a fucked up entropy. 

Our DecisionTree picks splits of the maximum **Information Gain**
 - First, calculate "Parents Entropy".
 - Second, look at the possible splits that **each column** gives, and caluculate each "Child Entropy".
 - Third, calculate each column's "Information Gain" to pick the largest. 
<img src="https://user-images.githubusercontent.com/31917400/40005096-1677691a-578f-11e8-8ef6-238d2b57f01d.jpg" />  
<img src="https://user-images.githubusercontent.com/31917400/38381197-f0b8b832-38fd-11e8-83da-db0be6a464ec.jpg" />  
<img src="https://user-images.githubusercontent.com/31917400/40007308-50a9a350-5794-11e8-80be-ff1449721e92.jpg" />  

### Hyperparameters for Decision Trees
 - `max_depth`
   - the largest length between the root to a leaf. A tree of maximum length k can have at most 2^k leaves(the very end).
   - Of course, too large depth very often causes overfitting.
 - `min_samples_leaf`(the very end)
   - a minimum for the number of samples we allow on each individual leaf.
   - This number can be specified as an integer, or as a float. If it's an integer, it's the number of minimum samples in the leaf. If it's a float, it'll be considered as the minimum percentage of samples on each leaf. For example, 0.1, or 10%, implies that a cut will not be allowed if in one of the leaves there is less than 10% of the samples on that node.
   - Of course, too small minimum samples per leaf results in overfitting.
 - `min_samples_split`
   - This is the same as the minimum number of samples per leaf, but applied on **any split** of a node.
 - `max_features`
   - Oftentimes, we will have too many features to build a tree. To speed up? we limit the number of features that one looks for in each split. 
<img src="https://user-images.githubusercontent.com/31917400/40022028-b2b645b0-57be-11e8-88e6-cde24608bbf4.jpg" />  
 
## RandomForests >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
<img src="https://user-images.githubusercontent.com/31917400/40016638-daa76b40-57ae-11e8-8086-077ad0a9f8b4.jpg" />  

What if we have so many columns? ---- Warning of Overfitting !!! How to solve ?
 - 1) Pick some of the columns randomly  
 - 2) Build a DecisionTree in those columns
 - 3) repeat 
 - 4) Let the trees vote ! 
   - When we have a new data-pt, let all the trees make a prediction and pick the one that appears the most. 

## EnsembleMethods >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
Take a bunch of models and join them together to get a better model
 - **Bagging**(Bootstrap Aggregating): Check each results and combine(average out or vote).
 - **Boosting**: A bit more elaborated than Bagging. Try hard to exploit the strength of each models then combine.
> Bagging
 - As our data is huge, we don't want to train many models on the same data. so we take random subsets of the data and train a week learner(one node DecisionTree) on each one of these subsets. 
 - Impose each result over the data and vote(as what two or more of them say..blue? it's blue.)
<img src="https://user-images.githubusercontent.com/31917400/40124901-772598c0-5921-11e8-8d90-2b69198d4f6f.jpg" />  

> Adaboosting
 - First, fit our first weak learner in order to maximize accuracy(or equivalently minimize the size of errors): Do no better than **3 errors** ! When it comes to the errors, it makes them bigger(punish them). 
 - Our second learner needs to fix on the **mistakes** that the first one has made, correctly classifying these points at any expense, then punish the points misclassified by itself. 
 - Our third learner needs to fix on the **mistakes** that the second one has made, correctly classifying these points at any expense, then punish the points misclassified by itself....we can go on and on..but let's say 3 is enough and we combine these learners. 
 - OVERALL
<img src="https://user-images.githubusercontent.com/31917400/40187029-708b6a12-59ee-11e8-813c-c55513de5e9d.jpg" />  
 
 - DETAIL
   - Assign an initial weight of '1' and before fit our first learner, minimize the size of errors, then minimize the SUM of weights of these errors by changing the weights of errors to: `correct_sum/incorrect_sum`, which will make these two correct, incorrect into the same SUM of the correct, incorrect. 
   - keep this going.
   - Here, notice `correct_sum/incorrect_sum` = `accuracy/(1-accuracy)` and we put it into `ln( )`, which is the final weight.
   - Our concern is whether the sums of these final-weights are `+ / -`. 
<img src="https://user-images.githubusercontent.com/31917400/40187295-1aee25bc-59ef-11e8-8177-ad2da67d5c1f.jpg" />  
<img src="https://user-images.githubusercontent.com/31917400/40186492-23f685ca-59ed-11e8-9ae7-7e41a8300071.jpg" />  

```
from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier()
model.fit(x_train, y_train)
model.predict(x_test)
```
When we define the model, we can specify the **hyperparameters**. In practice, the most common ones are:
 - `base_estimator`: The model(Here, DecisonTree..) utilized for the weak learners 
 - `n_estimators`: The maximum number of weak learners used.
```
from sklearn.tree import DecisionTreeClassifier
model = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth=2), n_estimators = 4)
```

--------------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## (C) Naive Bayes
 - For categoric data
 - 
 
> PREDICTION: when future emails come, we can combine these features to guess if they are spam or not. 
<img src="https://user-images.githubusercontent.com/31917400/39019180-5566df3e-4420-11e8-9f87-dad95387ce6c.jpg" />

 - Naive Bayes is an extension of the Bayes Theorem where we have more than one feature, with the assumption that each feature is independent of each other event..so Naive.
 - Library: sklearn.naive_bayes (Gaussian)
 - Example: Compute the accuracy of your Naive Bayes classifier. Accuracy is defined as the number of test points that are classified correctly divided by the total number of test points.
```
def NBAccuracy(features_train, labels_train, features_test, labels_test):
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()    ### create classifier ###
    clf.fit(features_train, labels_train)    ### fit the classifier on the training features and labels ###
    pred = clf.predict(features_test)    ### use the trained classifier to predict labels for the test features ###

    ### calculate and return the accuracy on the test data. ### 
    accuracy = clf.score(features_test, labels_test)
    return(accuracy)
    
    ### or we can use 'sklearn accuracy' ###
    from sklearn.metrics import accuracy_score
    print(accuracy_score(pred, labels_test))
```
It throws an accuracy of 88.4% which means 88.4% of the points are being correctly labelled by our classifier-'clf' when we use our test-set ! 

>__Bayes Rule:__ 
<img src="https://user-images.githubusercontent.com/31917400/40175347-b6a12bbe-59cf-11e8-9d9c-57fa7bfccced.JPG" />

*Semantically, what Bayes rule does is it **incorporates** some evidence from the test into our **prior** to arrive at a **posterior**.
 - Prior: Probability before running a test.
 - test evidence
 - Posterior: 
<img src="https://user-images.githubusercontent.com/31917400/40233114-a35e8b12-5a99-11e8-84ae-60f690a1fca6.jpg" />

*Algorithm of Naive Bayes
<img src="https://user-images.githubusercontent.com/31917400/40231554-75162b6c-5a93-11e8-9ce2-aec759b1c1fc.jpg" />

### Example: Text Forensic and Learning (ex. Whose email would it be ?)
<img src="https://user-images.githubusercontent.com/31917400/40237824-6cba9ece-5aa9-11e8-8468-16d3c0fdf0db.jpg" />


































-------------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------------
## (E) Perceptron Algorithm for "Neural-Network"
 - For categoric data (Y/N)
 - 
<img src="https://user-images.githubusercontent.com/31917400/39047770-10482214-4493-11e8-8103-03c5425c0534.jpg" />   
 
 - Our model has 'input data-pt', 'weights', 'bias'
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







































































