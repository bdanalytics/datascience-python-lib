"""
Lemons: 	mybenchmark

2013-12-24: Changed Output suffix to prefix

Problem:	Car is a good buy (0) or bad buy (1) - Classification
Data: 		Training & Prediction datasets in local directory
Model: 		Model Name
			List relevant features & importance along with importance term name
Outputs:	Prefix = "mybenchmark_"
"""

### 00. import modules
import datetime as tm
tm_start = tm.datetime.now()
print "[{0}] importing libraries ...".format(str(tm.datetime.now() - tm_start))

import sys
sys.path.append('/Users/bbalaji-2012/Documents/Work/DataScience/python')
import dsutils

import pandas as pd
import numpy as np

import copy
import pprint
pp = pprint.PrettyPrinter(indent=4)

from sklearn import dummy, tree, linear_model, ensemble, lda, qda
from sklearn import cross_validation, feature_selection
from sklearn import metrics

import matplotlib.pyplot as plt

exp_prefix = "mybenchmark_"
entity_name = 'vehicles'
predict_name = 'predict_vehicles'
rowkey_varname = 'RefId'
pred_varname = 'IsBadBuy'
random_varname = 'my_random'
mypred_varname = 'my_' + pred_varname

###	01.	load data
print "\n[{0}] loading data ...".format(str(tm.datetime.now() - tm_start))

entity = dsutils.myimport_data('lemon_training.csv', entity_name, pred_varname, index_col=0)
predict = dsutils.myimport_data('lemon_test.csv', entity_name + '.predict', pred_varname, index_col=0)
entity[random_varname] = np.random.randn(len(entity))
predict[random_varname] = np.random.randn(len(predict))

###	02.	clean data
print "\n[{0}] cleaning data ...".format(str(tm.datetime.now() - tm_start))
### 02.1	inspect data
### 02.2	fill missing data
###	02.3	drop cols that still contain NaNs
entity = dsutils.mydrop_na(entity, pred_varname)

###	03.	extract features
#print "\n[{0}] extracting features ...".format(str(tm.datetime.now() - tm_start))
### 03.1	convert non-numerical features to numeric features
### 03.2 	create feature combinations

###	04. transform features
print "\n[{0}] transforming features ...".format(str(tm.datetime.now() - tm_start))

### 04.1	collect all numeric features
numeric_feats = list(entity.describe().columns)

###	04.2	remove row keys & prediction variable
if rowkey_varname in numeric_feats:
	numeric_feats.remove(rowkey_varname)    #	Car rowkey

numeric_feats.remove(pred_varname)        #	prediction var

### 04.3	remove features that should not be part of estimation
numeric_feats.remove('BYRNO')        # 	Buyer # not significant for classification ???
numeric_feats.remove('VNZIP1')            #	ZIP code where care was purchased not significant for classification ???

features = numeric_feats

###	04.4	remove features / create feature combinations for highly correlated features
features = dsutils.myremove_corr_feats(entity, features, pred_varname, random_varname)

### 04.5	scale / normalize selected features for data distribution requirements in various models

###	05.	build training and test data
print "\n[{0}] creating train, validate & test sets ...".format(str(tm.datetime.now() - tm_start))

### 05.1	simple shuffle sample
### 05.2	stratified shuffle sample
train_entity, validate_entity, \
train_plus_validate_entity, test_entity = dsutils.mybuild_stratified_samples(entity, pred_varname)

### 06.	select models
print "\n[{0}] selecting models ...".format(str(tm.datetime.now() - tm_start))

### 06.1	select base models
###	06.1.1		regression models
###	06.1.2		classification models
sel_models = 	[dummy.DummyClassifier(strategy='most_frequent')
				,tree.DecisionTreeClassifier()
				,linear_model.LogisticRegression()
				,ensemble.RandomForestClassifier()
				,dsutils.myMultivariateGaussianClassifier()
				,lda.LDA()
				,qda.QDA()
				]
sel_models_index = 	['DummyClassifier'
					,'DecisionTreeClassifier'
					,'LogisticRegression'
					,'RandomForestClassifier'
					,'MultivariateGaussianClassifier'
					,'LDA'
					,'QDA'
					]
models = pd.DataFrame({'model': sel_models}, index=sel_models_index)
print "		selected models:"
#pp.pprint(models)
print "{0}".format(models.to_string(float_format=lambda x: '%0.4f' % x))

###	07.	design models
print "\n[{0}] designing models ...".format(str(tm.datetime.now() - tm_start))

### 07.1	select significant features
features = dsutils.myselect_significant_features(train_plus_validate_entity, entity,
												 features, pred_varname, random_varname, exp_prefix)

###	07.1.1		add back in key features even though they might have been eliminated
###	07.2	identify model parameters (e.g. # of neighbors for knn, # of estimators for ensemble models)

###	08.	run models
print "\n[{0}] running models ...".format(str(tm.datetime.now() - tm_start))
###	08.2	fit on stratified shuffled sample
models = dsutils.myfit_stratified_samples(models,
										  train_entity, validate_entity, train_plus_validate_entity,
										  test_entity,
										  features, pred_varname, mypred_varname,
										  exp_prefix)

### 08.3    fit on cross-validated samples

"""
###	09.	test model results
print "\n[{0}] testing model results ...".format(str(tm.datetime.now() - tm_start))

###	09.1	collect votes from each cross-validation fold for each model
### 09.2	collect votes from each model
### 09.3 	export test data for inspection

###	10.	predict results for new data
print "\n[{0}] running models on prediction data ...".format(str(tm.datetime.now() - tm_start))
###	10.1	run models with data to predict
###	10.2	collect votes from each cross-validate for each model
### 10.3	collect votes from each model

###	11.	export results
print "\n[{0}] exporting results ...".format(str(tm.datetime.now() - tm_start))
"""