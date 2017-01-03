---
layout: post
title: "Testing Readability with a Bunch of Text"
date: 2012-05-22
excerpt: "A ton of text to test readability."
tags: [sample post, readability, test]
comments: true
---

```python
	def s():
	dkfl 
```


# Naive Bayes Classifier

## Model Summary

The Naive Bayes Classifier is an algorithm based on Baye's Theorem, which can be represented as:

![Title](extras/bayes_rule.png)

where:
 * P(c|x) is the posterior probability of class (c, or outcome variable) given predictor (x, or attributes or features).
 * P(x|c) is the likelihood/probability of the predictor occuring given class.
 * P(c) is the prior probability of class (overall probability that it occurs).
 * P(x) is the prior probability of predictor (also known as 'evidence' in Bayesian probability terminology).

The conditional probabilities are then multiplied across all features for each class, and the class with the highest probability is chosen. It is very important to note that a key assumption of this model is that the features are independent, which is where the _naive_ name is derived.

## Theoretical Example

Let us predict the probability that a phone will explode given that it's a Samsung Note, P(c|x). If 1% of all phones explode, 25% of all phones are Samsung Note, and 75% of all exploding phones are Samsung Note, then P(c|x) is ((.75)(.25))/(.01) = 18.75%.

## Pros & Cons

Pros:
 - It is a very fast classifier.
 - It performs well with categorical input variables compared to numerical ones.

Cons:
 - If a categorical variable has a value that is observed in the Test data set that didn't exist in the Training data set, the model assumes a zero probability and cannot make a prediction. This can be fixed using the Laplace estimation smoothing technique.
 - The model is known to be a bad predictor, meaning outputs from predict_proba are worthless.
 - The assumptions of independent features or normal distribution in the Gaussian NB model are very strong.
     - Note: it's been shown that even if the assumption of independent features is violated, the performance of the classifier will not be noticeably negatively affected.
  
## Applications

 - Real time prediction: NB is a very fast classifier. 
 - Multi-class prediction
 - Text Classification
 - Recommender Systems: NB is often used with Collaborative Filtering to build good recommender systems.

## Model Types

There are three primary Naive Bayes models, each determined by the kind of feature variables we're working with: 
 - Gaussian NB: assumes that the features are numerical follow a normal distribution.
 - Multinomial NB: assumes that the feature variables are discrete counts, and is often used in text classification that analyzes word counts.
 - Bernoulli NB: assumes that the features are binary variables, and is often used in text classification that looks for the presence of a given word.


## Bernoulli Classification Coding Example

In this example, we will use the Bernoulli classification model to predict the _Gender_ of an ASU PSC applicant using the _Military_ and _Ethnicity_ fields. 


```python
import os
import numpy as np
import pandas as pd
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import sys
```

First we load the data (available on Kaggle):


```python
data = pd.read_csv('Data/titanic_train.csv')
```


```python
data.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



Judging by the quick inspection of columns, we cna take Pclass, Sex, and Embarked as our categorical predictors, which we will turn into binary variables.


```python
X = pd.concat([pd.get_dummies(data[['Sex', 'Embarked']]),pd.get_dummies(data['Pclass'])], axis=1)
y = data['Survived']
```

Split the data into train and test sets:


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .33, random_state = 101)
```

Run the model using a Bernoulli Naive Bayes:


```python
nb = BernoulliNB()
nb.fit(X_train, y_train)

ber_p = nb.predict_proba(X_train)
```

Evaluate the model on the test dataset:


```python
yhat = nb.predict(X_test)

ber_pt = nb.predict_proba(X_test)

print confusion_matrix(y_test, yhat)
print '\n', classification_report(y_test, yhat)
```

    [[138  31]
     [ 35  91]]
    
                 precision    recall  f1-score   support
    
              0       0.80      0.82      0.81       169
              1       0.75      0.72      0.73       126
    
    avg / total       0.78      0.78      0.78       295
    
    

## Gaussian NB Coding Example  

In this example, we will look implement the Gaussian Naive Bayes on the numerical features in our data set (Age, Fare, and Number of Siblings).  
First, we will fill in the Age nulls with the overall averages.


```python
data['Age'].fillna(data['Age'].mean(), inplace=True)
```

Split the data into train and test sets.


```python
X = data[['Age', 'Fare', 'SibSp']]
y = data['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .33, random_state = 101)
```

Run the Gaussian NB Model:


```python
gnb = GaussianNB()
gnb = gnb.fit(X_train, y_train)

gnb_p = gnb.predict_proba(X_train)
```

Evaluate the model on the test dataset:


```python
yhat = gnb.predict(X_test)

gnb_pt = gnb.predict_proba(X_test)

print confusion_matrix(y_test, yhat)
print '\n', classification_report(y_test, yhat)
```

    [[161   8]
     [ 92  34]]
    
                 precision    recall  f1-score   support
    
              0       0.64      0.95      0.76       169
              1       0.81      0.27      0.40       126
    
    avg / total       0.71      0.66      0.61       295
    
    

## Combining the Bernoulli and Gaussian Models  

In order to maximize the predictive power of our model, we want to combine the predicted probabilities of the Gaussian and Bernoulli models and fit a new Gaussian model on top of the new data.


```python
X_comb = np.hstack([ber_p, gnb_p])
x_comb_test = np.hstack([ber_pt, gnb_pt])
```


```python
gnb_new = GaussianNB()

gnb_new = gnb_new.fit(X_comb, y_train)

yhat = gnb_new.predict(x_comb_test)
```

Evaluate the model on the test dataset:


```python
print confusion_matrix(y_test, yhat)
print '\n', classification_report(y_test, yhat)
```

    [[140  29]
     [ 38  88]]
    
                 precision    recall  f1-score   support
    
              0       0.79      0.83      0.81       169
              1       0.75      0.70      0.72       126
    
    avg / total       0.77      0.77      0.77       295
    
    

The results of the combined model indicate that the poor performing Gaussian model dragged down our relatively well-performing Bernoulli model. These findinds are consistent with our knowledge that Naive Bayes works better with categorical predictors than with numerical predictors, and that the predicted probabilities do not yield good results generally.

## Alternative Approach to Gaussian Model  

As we have demonstrated, the Gaussian model doesn't offer much of an improvement over our Bernoulli model. However, that's not to say that the numeric features cannot contribute to the Bernoulli model. Let's see what happens when we convert the numeric features into categorical labels.


```python
X = pd.concat([pd.get_dummies(data[['Sex', 'Embarked']]),pd.get_dummies(data['Pclass'])], axis=1)

for i in ['Age', 'Fare', 'SibSp']:
    try:
        data[i+'_label'] = pd.qcut(data[i],4)
    except (ValueError):
        data[i+'_label'] = pd.cut(data[i],4)
    X = pd.concat([X, pd.get_dummies(data[i+'_label'],prefix=i, prefix_sep='_')], axis=1)
```

Now let's see how the Bernoulli model performs:

Split the data into train and test sets:


```python
y = data['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .33, random_state = 101)
```

Run the model using a Bernoulli Naive Bayes:


```python
nb = BernoulliNB()
nb.fit(X_train, y_train)

ber_p = nb.predict_proba(X_train)
```

Evaluate the model on the test dataset:


```python
yhat = nb.predict(X_test)

ber_pt = nb.predict_proba(X_test)

print confusion_matrix(y_test, yhat)
print '\n', classification_report(y_test, yhat)
```

    [[141  28]
     [ 38  88]]
    
                 precision    recall  f1-score   support
    
              0       0.79      0.83      0.81       169
              1       0.76      0.70      0.73       126
    
    avg / total       0.78      0.78      0.77       295
    
    

Looks like the is very comparable to our initial model. 

## Additional Resources

 - http://scikit-learn.org/stable/modules/naive_bayes.html#naive-bayes
 - https://en.wikipedia.org/wiki/Naive_Bayes_classifier
 - http://www.analyticsvidhya.com/blog/2015/09/naive-bayes-explained/

