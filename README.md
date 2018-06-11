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
 - The dependent variable must be categorical, and the explanatory variables can take any form.
 - But before fitting on the data, we need to convert the categorical predictors into the numeric(using `pd.get_dummies(series)[which level is '1'?]`). This is called:
   - `One-Hot Encoding` when there is no ordinal levels (red, blue,..=> check(1), uncheck(0),..)
   - `Integer Encoding` when there is ordinal levels (first, second,..=> 1,2..) 
   - http://pbpython.com/categorical-encoding.html

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

 - Unlike SVM using a kernel trick, **DecisionTree** use a trick that lets a linear-DecisionSurf do Non-Linear-Decision making.
 - When making Decision Trees, we ask questions: On what features do we make our decisions on? What is the threshold for classifying each question into a yes or no answer? By adding an additional question, we can greater define the Yes and No classes ! 
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

> Bayes Rule: 
<img src="https://user-images.githubusercontent.com/31917400/40175347-b6a12bbe-59cf-11e8-9d9c-57fa7bfccced.JPG" />

*Semantically, what Bayes rule does is it **incorporates** some evidence from the test into our **prior** to arrive at a **posterior**.
 - Prior: Probability before running a test.
 - test evidence
 - Posterior: 
<img src="https://user-images.githubusercontent.com/31917400/40233114-a35e8b12-5a99-11e8-84ae-60f690a1fca6.jpg" />

*Algorithm of Naive Bayes
<img src="https://user-images.githubusercontent.com/31917400/40231554-75162b6c-5a93-11e8-9ce2-aec759b1c1fc.jpg" />

### Ex) Text Forensic and Learning (ex. Whose email would it be ?)
<img src="https://user-images.githubusercontent.com/31917400/40242300-345d0664-5ab5-11e8-9a71-0daeb1a14317.jpg" />

### Ex) Multiple Evidences(test results)
<img src="https://user-images.githubusercontent.com/31917400/40252829-85c1ab3e-5ad5-11e8-98c6-bfe4e170fe22.jpg" />

Spam detection is one of the major applications of Machine Learning in the interwebs today. Pretty much all of the major email service providers have spam detection systems built in and automatically classify such mail as 'Junk Mail'.

> What are spammy messages?
Usually they have words like 'free', 'win', 'winner', 'cash', 'prize' and the like in them as these texts are designed to catch your eye and in some sense tempt you to open them. Also, spam messages tend to have words written in all capitals and also tend to use a lot of exclamation marks. To the recipient, it is usually pretty straightforward to identify a spam text and our objective here is to train a model to do that for us! Being able to identify spam messages is a **binary classification** problem as messages are classified as either 'Spam' or 'Not Spam' and nothing else. This project has been broken down in to the following steps:
 - Step 0: Introduction to the Naive Bayes Theorem
 - Step 1.1: Understanding our dataset
 - Step 1.2: Data Preprocessing
 - Step 2.1: Bag of Words(BoW)
 - Step 2.2: Implementing BoW from scratch
 - Step 2.3: Implementing Bag of Words in scikit-learn
 - Step 3.1: Training and testing sets
 - Step 3.2: Applying Bag of Words processing to our dataset.
 - Step 4.1: Bayes Theorem implementation from scratch
 - Step 4.2: Naive Bayes implementation from scratch
 - Step 5: Naive Bayes implementation using scikit-learn
 - Step 6: Evaluating our model
 - Step 7: Conclusion



--------------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## (D) Support Vector Machine
 - For categoric data
 - For numeric data
> PREDICTION: face detection, handwriting recognition, time series, stock value prediction
<img src="https://user-images.githubusercontent.com/31917400/39476600-673208d2-4d54-11e8-8f73-18871433f89c.jpg" />

SVM is a set of supervised learning methods used for 
 - classification
 - regression  
 - **outliers detection**
SVMs "doesn't work well with lots and lots of noise, so when the classes are very overlapping, you have to count independent evidence.

It is not a probabilistic model; i.e., it does not postulate a probability distribution and thus does not assume any randomness. It merely tries to draw a simple line(or plane or hyperplane in higher dimensions) to separate the data points into two parts. That's all. Note that the dataset contains labeled data.

One difficulty was that oftentimes the classifier(the separating 'line' or 'hyperplane') cannot be defined linearly, which means it's not actually a straight line or plane that separates the two sets. It should rather be a wavy curve or surface. So what do we do? We lift the feature space to a higher or possibly an infinite dimensional space so that a linear classifier is possible. This is called the kernel trick. This is what the support vector machine does.

Now applying this to a regression problem, linear regression could be described as an attempt to draw a line (or similarly plane or hyperplane in higher dimensions) that minimizes the error(or the loss function). Therefore, if we choose different loss functions, the regression line(or plane, hyperplane) changes. When the feature space seemingly isn't best served by a simple line or plane but rather calls for something wavy as seen in the classification problem, instead of approximating the wavy object, we again use the kernel trick to lift the feature space into a higher dimension. In this task, the output is a real value.

### In SVM, tuning the parameters can be a lot of work, but GridCV, a great sklearn tool that can find an optimal parameter tune almost automatically.

Naive Bayes is great for 'text'. It’s faster and generally gives better performance than an SVM. Of course, there are plenty of other problems where an SVM might work better. Knowing which one to try when you’re tackling a problem for the first time is part of the art of ML. 

Pros & Cons
 - > The advantages of support vector machines are:
   - Effective in cases where number of dimensions is greater than the number of samples.
   - Uses a subset of training points in the decision function called `support vectors`, so it is also memory efficient.
   - Versatile: different **Kernel functions** can be specified for the decision function(Common kernels are provided, but it is also possible to specify custom kernels). 
   - Using a **kernel trick**, Linear DecisionSurf -> NonLinear DecisionSurf    

 - > The disadvantages of support vector machines include:
   - If the number of features is much greater than the number of samples, avoid **over-fitting** in choosing Kernel functions and **regularization term** is crucial.
   - SVMs do not directly provide probability estimates, these are calculated using an expensive **five-fold cross-validation**.
<img src="https://user-images.githubusercontent.com/31917400/35055161-61987186-fba6-11e7-8c97-b66617e8161c.jpg" width="750" height="150" />

Margine is a maximum distance to each nearest point. The separating line should be most robust to classification errors. The margine aims to maximizes the robustness of the result....As Much Separation b/w two classifications as possible. 
 - The perceptron algorithm is a trick in which we started with a random line, and iterated on a step in order to slowly walk the line towards the misclassified points, so we can classify them correctly. However, we can also see this algorithm as an algorithm which minimizes an error function. 
<img src="https://user-images.githubusercontent.com/31917400/40259702-298552a2-5aef-11e8-9820-21406a2e0386.jpg" />

Error (Margin Error + Classification Error)
 - We punish the smaller margin..(just like punishing the model complexity in the L2_regularization of LinearModel). We love the larger margin.
 - We want to minimize the total error (or error function)
<img src="https://user-images.githubusercontent.com/31917400/40268051-8b7dc1b4-5b5e-11e8-8604-bb5e4468e452.jpg" />
<img src="https://user-images.githubusercontent.com/31917400/40268052-8f949516-5b5e-11e8-8efc-d44acfa0eee3.jpg" />

```
import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
import copy

import sys
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData

features_train, labels_train, features_test, labels_test = makeTerrainData()
```
In sklearn.svm, `SVC()`, `NuSVC()`, `LinearSVC()` accept slightly different sets of parameters and have different mathematical formulations, but take as input two arrays: 
 - an array **X** of size `[n_samples, n_features]`holding the training samples 
 - an array **y** of class labels (strings or integers), size `[n_samples]`
 - Library: sklearn.svm 
 - Example: 
```
from sklearn.svm import SVC
# clf = SVC(kernel="linear") #
# clf = SVC(kernel='poly', degree=4, C=0.1) #
# clf = SVC(kernel='rbf', gamma= ) #

X = features_train
y = labels_train
clf.fit(X, y)

pred = clf.predict(features_test)
```
Accuracy ?
```
from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)

def submitAccuracy():
    return acc
```
## Non-Linear SVM
<img src="https://user-images.githubusercontent.com/31917400/39048647-2962d1ba-4496-11e8-82ee-b87365d27b07.jpg" />  

Introducing New Features 'Z' or 'transformed X or Y' causes 'hyperplane.' Z is non-negative because it's a distance from the origin. 
<img src="https://user-images.githubusercontent.com/31917400/35122461-b94d14f4-fc96-11e7-9e22-1e3a76e58e16.jpg" />

Introducing poly tool box. Select terms to create the decision surf.
<img src="https://user-images.githubusercontent.com/31917400/40273507-fc70f328-5bb8-11e8-948c-4b087934f4d9.jpg" />  

**Kernel Trick:** There are functions taking a low dimensional given 'input space' and the added 'feature space' then map it to a very high dimensional space - Kernel function (Linear, poly, rbf, sigmoid). It makes the separation then takes the solution and go back to the original space. It sets the dataset apart where the division line is non-linear.
<img src="https://user-images.githubusercontent.com/31917400/35122799-e8106e2a-fc97-11e7-8872-43e13edacfd9.jpg" width="500" height="100" />

rbf (radial basis func) kernel: 
 - hill & valley
 - find a place where a line intersecting the mountain range and project every pt down, then we have a boundary given by the vertical cut. But how we build the mountain range and how to locate red pt in highlands and blue pt in lowlands ?  

parameters (degree, C, Gamma)
 - **C:** The 'gamma' parameter actually has no effect on the 'linear' kernel for SVMs. The key parameter for 'linear kernel function' is "C". The C parameter **trades off misclassification of training examples against simplicity of the decision surface**. A low C makes the decision surface smooth, while a high C aims at classifying all training examples correctly by giving the model freedom to select more samples as support vectors - wiggling around individual data pt...
 - C is a constant that attaches itself to the classification error. If we have large C, then the error is mostly the classification error so we focus more on correctly classifying all the points than in finding a good margin. When C is small, the error is mostly a margin error. 
<img src="https://user-images.githubusercontent.com/31917400/40270444-832ce66c-5b85-11e8-8512-21274e8a962c.jpg" />

 - **Gamma:** This parameter in **rbf** defines **how far the influence of a single data pt reaches**, with low values (widey mountain) meaning ‘far’ and high values (pointy mountain) meaning ‘close’. The gamma parameters can be seen as the inverse of the radius of influence of samples selected by the model as support vectors. High gamma just like me..only thinking of sth right in my face. 
 - When gamma is very small, the model is too constrained and cannot capture the complexity or “shape” of the data. The region of influence of any selected support vector would include the whole training set. The resulting model will behave similarly to a linear model with a set of hyperplanes that separate the centers of high density of any pair of two classes. If gamma is too large, the radius of the area of influence of the support vectors only includes the support vector itself and no amount of regularization with C will be able to prevent overfitting. 
<img src="https://user-images.githubusercontent.com/31917400/35127560-923ca17c-fcaa-11e7-81ca-e4db864ccc96.jpg" />  

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
 - Perceptron refers a combination of nodes (here, Linear_function_Node + Step_function_Node)
 - Application example: Perceptron can be a logical operator: AND, OR, NOT, XOR...
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
 - [**What if it's impossible to build the decision surface ?**]
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
        y_hat = prediction(X[i],W,b) ## it's | ---
        if y[i]-y_hat == 1:  ## FalseNegative
            W[0] += learn_rate*X[i][0]
            W[1] += learn_rate*X[i][1]
            b += learn_rate
        elif y[i]-y_hat == -1:  ## FalsePositive
            W[0] -= learn_rate*X[i][0]
            W[1] -= learn_rate*X[i][1]
            b -= learn_rate
    return W, b
```
This function runs the perceptron algorithm repeatedly on the dataset, and returns a few of the boundary lines obtained in the iterations for plotting purposes.
> Play with the **learning rate** and the **num_epochs**.
 - 'boundary_lines' are the solution lines that get plotted below.
 - In each epoch, we apply the perceptron step.
<img src="https://user-images.githubusercontent.com/31917400/41202123-c0a19c0c-6cbc-11e8-94e2-76d73f9a297b.jpg" />
<img src="https://user-images.githubusercontent.com/31917400/41205313-1c04212e-6ce9-11e8-8b1d-c9bf69912d72.jpg" />

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

## Non-Linear DecisionSurf and perceptrons

### Concept_01. **Error-Function**(Gradient_Descent Method)
It tells us how far we are from the solution(it's a distance).
 - It should be continuous!
 - It should be differentiable! (just like minimizing SSE in linear model.)
<img src="https://user-images.githubusercontent.com/31917400/41206106-c62ad73a-6cf6-11e8-8307-d38aeda8113a.jpg" />

> Move from the discrete to the continuous!
<img src="https://user-images.githubusercontent.com/31917400/41206423-b5992a52-6cfb-11e8-911f-b406ac6c3f5a.jpg" />

Two perceptrons
 - Note the change in activation function(from Step to Sigmoid). Instead of returning 0/1, the new percaptron offer each probability!
<img src="https://user-images.githubusercontent.com/31917400/41206510-138f1800-6cfd-11e8-8cec-63a6233c6ff0.jpg" />

### Concept_02. Multi-class Classification

**1. Softmax Function**

When the problem has 3 or more classes ? How to turn all scores(WX+b, which is a linear function) into probabilities?(of course, Sigmoid)
 - Note that scores often can be negative, and we need to calculate probability. See 'exp() of Sigmoid' can turn them into positive. 
<img src="https://user-images.githubusercontent.com/31917400/41224711-06ad2c7c-6d65-11e8-8264-0f966ac381b3.jpg" />

 - Let's say we have 'n' classes and our linear model(WX+b) gives us the score: Z_1...Z_n, each score for each class. Let's turn them into probabilities. Takes as input a list of numbers(scores), and returns the list of values(possibilities) given by the softmax function.
```
def softmax(L):
    expL = np.exp(L)
    S_expL = sum(expL)
    result=[]
    for i in expL:
        result.append(i/S_expL)
    return(result)
    
def softmax(L):
    expL = np.exp(L)
    return(np.divide(expL, sum(expL)))    
```
**2. One hot encoding**

What if some input data is not numerical?   
<img src="https://user-images.githubusercontent.com/31917400/41227645-7153369a-6d6d-11e8-9e91-5b637e992979.jpg" />

**3. MaximumLikelihood** to improve our model

Want to calculate **probability the four points are of the colors** that they **actually** are. We assume the colors of the points are independent events, then the probability for the **whole arrangement** is the product of the probabilities of the four points. If the model is given by these probability spaces, then the **probability that the points are of this colors** offers the clue of which model is better. 
<img src="https://user-images.githubusercontent.com/31917400/41233369-28e09cd6-6d81-11e8-947d-11ba772b9e33.jpg" />

 - So how to maximize the probability? 
 - So how to minimize the Error-Function? 
 - Can we obtain an error-Function from the probability? Maximized probability can yield the minimised Error-Function?
 - What if the number of datapoints are astronomical? Then producting is not a good idea. We need a log-function that turns products into sums...and remember..when input is ranged from 0 to 1, the logarithm gives negative. And this is the Entropy function. 

**4. Cross-Entropy**

If I have a bunch of events and probabilities, Cross-Entropy says **how likely those events happen based on the probabilities**. If it's highly likely, then we have a small Cross-Entropy. If it's unlikely, we have a large Cross-Entropy. 
 - A good model gives a low cross-entropy and a bad model gives a high cross-entropy. So our goal has changed: 
   - **Minimize the Cross Entropy!**
<img src="https://user-images.githubusercontent.com/31917400/41236233-86a74a52-6d88-11e8-801e-6eeccc2afdec.jpg" />
<img src="https://user-images.githubusercontent.com/31917400/41238775-acc5f0b6-6d8e-11e8-882a-22c402206915.jpg" />
<img src="https://user-images.githubusercontent.com/31917400/41238777-af71af9e-6d8e-11e8-98d6-6db510dc2570.jpg" />

 - Cross Entropy is a connection between probabilities and error functions.









































































