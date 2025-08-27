# Supervised-machine-learning-classifiers

I have used 3 CSV data files here :
“train.csv” contains 8,000 rows and 11 columns. The first column ‘y’ is the output variable with 4 classes: 0, 1, 2, 3. The remaining 10 columns contain input features: x1, …, x10. 
“test.csv” contains 2,000 rows and 11 columns. The first column ‘y’ is the output variable with 4 classes: 0, 1, 2, 3. The remaining 10 columns contain input features: x1, …, x10.
“new.csv” contains 30 rows and 11 columns. The first column ‘ID’ is an identifier for 30 unlabeled samples. The remaining 10 columns contain input features: x1, …, x10.

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
