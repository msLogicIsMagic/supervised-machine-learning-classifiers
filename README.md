# Supervised Machine Learning Classifiers

Supervised machine learning is one of the three common machine learning paradigms, the other two being, unsupervised and reinforcement learning. In this porject, I am focusing on supervised machine learning. Supervised machine learning learns from a set of labeled training examples so that it can predict output labels for inputs that have not be used for training. Each labeled training example is in the form of a pair (x,y) where x represents the values of the input features and y is the output label. The algorithm learns a function f that maps the input to an output, f(x) = y. This function is known as the supervised learning model.

A classification problem is a prediction problem where the output y can only assume a finite number of values. Each value that can be assumed by y is called a class.

Issues to consider for supervised models : 

Overfitting : If a model has significantly higher prediction accuracy for the training examples than it has for examples that has not used for training, then the model is said to be overfit. When error occurs while predicting the output for a sample input that was not used for training, it is known as a generalization error. When developing a model, the goal should be to minimize generalization error. One way to ensure that overfitting does not occur is the minimize the noise level in training data.

Methods to prevent overfitting : 
1. Use simpler models
   		Risk of overfitting increases as the model becomes more complex. Occam's razor principle states that of all the models that can reliably predict the outputs for the training samples, the simplest model must be chosen in order to reduce generalization error.
2. Reduce the influence of individual training examples
   		A noisy training data misleads the learning algorithm into learning relationships that dont exist. This is because noisy training data do not represent the true underlying rwlationship between inputs and output. We should make sure that the individual training examples do not have a disproportionate influence on the model.

## Model Evaluation

When training and testing a supervised machine learning classifier, the labeled sample data is split into 2 parts : tarining data and testing data. The common split ratio is 70% for training and 30% for testing. This is the most widely used and often recommended split which provides a substantial amount of data for the model to learn from while also reserving a good amount for independent testing and evaluation.

For the pupose of this project, I have used 3 CSV data files here :
“train.csv” contains 8,000 rows and 11 columns. The first column ‘y’ is the output variable with 4 classes: 0, 1, 2, 3. The remaining 10 columns contain input features: x1, …, x10. 
“test.csv” contains 2,000 rows and 11 columns. The first column ‘y’ is the output variable with 4 classes: 0, 1, 2, 3. The remaining 10 columns contain input features: x1, …, x10.
“new.csv” contains 30 rows and 11 columns. The first column ‘ID’ is an identifier for 30 unlabeled samples. The remaining 10 columns contain input features: x1, …, x10.

As you can see, the labeled sample data is divided in a 80-20 split which provides similar benefit as a 70-30 split. This kind of split is especially beneficial for more complex models or when there is a large dataset.

It is important to note that training samples should never be used to evaluate a model. 

## Hyper Parameter Tuning

To prevent overfitting, we experiment and adjust the hyper parameters of a model. This is known as hyper parameter tuning. When conducting hyper parameter tuning, we should not use test data. A portion of the training data is used for this purpose. Samples used for hyper parameter tuning are called validation sample. 

## k-fold Cross Validation

k-fold cross validation is the method used to tune the hyper parameters of a model and evaluate models. 

Step 1 : Suppose there are n labeled training samples. This is partitioned into k subsets where each partition contains n/k samples.
Step 2 : For each partition i, 
			a) Train the model on samples in the k-1 partitions, and
   			b) Calculate and collect the evaluation metrics using the samples in partition i 
Step 3 : Calculate the average value of the evaluation metrics from the k trials.

Benefits of k-fold cross validation are :
1. The training sample is used efficiently.
2. Since multiple trials are done, it is able to achieve more reliable results.

I have used 4-fold cross-validation with the 8,000 labeled exampled from “train.csv” to identify a classifier that achieves mean cross-validation accuracy of at least 0.97. Here, I have experimented with several of the Scikit-Learn classifiers such as DecisionTreeClassifier, RandomForestClassifier, ExtraTreesClassifier, KNeighborsClassifier, GaussianNB, SVC, LogisticRegression, and MLPClassifier. 

Given below are the mean cross-validation accuracy achieved with different classifiers :

GaussianNB : 0.9344

DecisionTreeClassifier : 0.9126

RandomForestClassifier : 0.9667

ExtraTreesClassifier : 0.9737

KNeighborsClassifier : 0.9774

LogisticRegression : 0.9646

SVC : 0.9794

MLPClassifier : 0.9709

Here, we can see that 4 classifiers are able to achieve a mean cross-validation accuracy of atleast 0.97 : ExtraTreesClassifier, KNeighborsClassifier, SVC and MLPClassifier.

We will consider the 2 models : SVC and KNeighborsClassifier with the highest mean cross-validation among them for further analysis. 

Then I experimented with different values of hyper-parameter for the better performing classifiers to obtain a good set of hyper-parameter values. This was then used to select the best performing model.

Selected model with non-default hyper-parameter values: Support Vector Classifier

Non-default hyper parameters :
	C: 10
	break_ties: False
	cache_size: 200
	class_weight: None
	coef0: 0.0
	decision_function_shape: ovr
	degree: 3
	gamma: scale
	kernel: rbf
	max_iter: -1
	probability: False
	random_state: None
	shrinking: True
	tol: 0.001
	verbose: False

Mean cross-validation accuracy: 0.9805 (rounded to 4 decimal places)
