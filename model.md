<h1 style="text-align: center;" markdown="1">Data Science for fraud detection</h1>

<div style="text-align: center;" markdown="1"><font size="5">
A complete predictive modeling project in Python
</font>  
<font size="4">
Part III: Model development
</font>
</div>

_____


## Articles

[Part I: Preprocessing and exploratory analysis
](article.html)  
[Part II: Feature engineering
](feature_eng.html)  
[Part III: Model development
](model.html)

To follow the following article without any trouble, I would recommend to
start with the beginning.
___

# How does predictive modeling work

## Keep the terminology in mind
This is important to understand the principles and
sub-disciplines of machine learning. We are trying to predict a specific
**output**, our information of interest, which is the category of bank note
we observe (genuine or forged).
This task is therefore labelled as **supervised learning**, as opposed to
**unsupervised learning** which consists of finding patterns or groups from
data without a priori identification of those groups.  

Supervised learning can further be labelled as **classification** or
**regression**, depending on the nature of the outcome, respectively
categorical or numerical. It is essential to know because the two disciplines
don't involve the same models. Some models work in both cases but their expected
behavior and performance would be different. In our case, the outcome is
categorical with two levels.  

## How does classification work?

Based on a subset of the data, we train a
model, so we tune it to minimize its error on these data. To make a parallel
with Object-Oriented Programming, the model is an **instance** of the
class which defines how it works. The attributes would be its parameters and
it would always have two methods (functions usable only from the object):  
* **train** the model from a set of observations (composed of predictive
    variables and of the outcome)
* **predict** the outcome given some new observations
Another optional method would be **adapt** which takes new training data and
adjusts/corrects the parameters. A brute-force way to perform this is to call
the train method on both the old and new data, but for some models a more
efficient technique exists.  

## Independent evaluation
A last significant element: we mentioned using only a subset of the data to
train the model. The reason is that the performance of the model has to be
evaluated, but if we compute the error on the training data, the result will
be biased because the model was precisely trained to minimize the error on this
training set. So the evaluation has to be done on a separated subset of the
data, this is called **cross validation**.

# Our model: logistic regression

Even though this is often not the most efficient algorithm on real data,
it is visually easy to understand and simple to implement from scratch.
Plus, it covers a central topic in data science, optimization.
The underlying reasoning is the following:  
The logit function of the probability of a level of the classes is
linearly dependent on the predictors. Why do we need the logit function here?
Well technically, a linear regression could be fitted with the class as output
(encoded as 0/1) and the features as predictive variables. However, for some
values of the predictors, the model would yield outputs below 0 or above 1.
The logistic function **equation** yields an output between 0 and 1 and
is therefore well suited to model a probability.



## Required parameters

As you noticed in the previous explanation, the model takes a vector of
parameters which correspond to the weights of the different variables.
