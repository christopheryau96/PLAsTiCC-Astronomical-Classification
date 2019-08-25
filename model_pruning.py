import pandas as pd
import numpy as np
import csv
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters
from sklearn.model_selection import	GridSearchCV
from tsfresh.feature_extraction import MinimalFCParameters

sub_pb = pd.read_csv('../modules/cs342/Assignment2/training_set.csv',header=0,usecols=[0,1,2,3],names=['object_id','mjd','passband','flux'])
sub_meta = pd.read_csv('../modules/cs342/Assignment2/training_set_metadata.csv',header=0,usecols=[0,11],names=['object_id','target'])


if __name__ == "__main__":
	
	fc_parameters={
		'abs_energy': None,
		'cid_ce':[{'normalize':False}],
		'energy_ratio_by_chunks':[{'num_segments':3,'segment_focus':0},{'num_segments':3,'segment_focus':1},{'num_segments':3,'segment_focus':2}]}
		
	kind_to_fc_parameters = {'0.0':fc_parameters,'1.0': fc_parameters,'2.0': fc_parameters,'3.0': fc_parameters,'4.0' : fc_parameters,'5.0': fc_parameters}

	train_features4=extract_features(sub_pb,column_id='object_id',column_value='flux',column_sort='mjd',column_kind='passband',
	default_fc_parameters=MinimalFCParameters())
	impute(train_features4)
		
	train_features5=extract_features(sub_pb,column_id='object_id',column_value='flux',column_sort='mjd',column_kind='passband',
	default_fc_parameters=fc_parameters,kind_to_fc_parameters=kind_to_fc_parameters)
	impute(train_features5)
	
	#Task5 model pruning
	training5_X = train_features5
	training5_Y = sub_meta['target']
	
	mlp5=MLPClassifier(random_state=1)
	rfc5=RandomForestClassifier(random_state=1)
	
	#params_rf = {'n_estimators': [300,400,500],'min_samples_leaf':[0.1,0.2],'max_features':['log2','sqrt'],'max_depth':	[3,4,5,6]}
	#grid_rf = GridSearchCV(estimator=rfc5,param_grid=params_rf,cv=5,scoring='neg_mean_squared_error')
	#grid_rf.fit(training5_X,training5_Y)
	#print grid_rf.best_params_
	
	params_mlp={'learning_rate': ["constant", "invscaling",  "adaptive"],
	'hidden_layer_sizes': [(30),(38),(20),(25),(34)],
	'alpha': [1e-5,1e-4,1e-3],
	'activation': ['logistic', 'relu'],
	'solver': ['adam','sgd']}
	grid_mlp=GridSearchCV(estimator=mlp5,param_grid=params_mlp,cv=5)
	grid_mlp.fit(training5_X,training5_Y)
	print grid_mlp.best_params_
	

	#Task4 model pruning
	training4_X = train_features4
	training4_Y = sub_meta['target']
	
	mlp4 = MLPClassifier(random_state=1)
	
	rfc4 = RandomForestClassifier(random_state=1)

	#parameter tuning for mlp
	params_mlp={'learning_rate': ["constant", "invscaling", "adaptive"],
	'hidden_layer_sizes': [(30),(38),(20),(25),(34)],
	'alpha': [1e-5,1e-4,1e-3],
	'activation': ['logistic', 'relu'],
	'solver': ['adam','sgd']}
	grid_mlp=GridSearchCV(estimator=mlp4,param_grid=params_mlp,cv=5)
	grid_mlp.fit(training4_X,training4_Y)
	print grid_mlp.best_params_
	
	#parameter tuning for randomforest
	params_rf = {'n_estimators': [300,400,500],'min_samples_leaf':[0.1,0.2],'max_features':['log2','sqrt'],'max_depth':	[3,4,5,6]}
	grid_rf = GridSearchCV(estimator=rfc4,param_grid=params_rf,cv=5,scoring='neg_mean_squared_error')
	grid_rf.fit(training4_X,training4_Y)
	print grid_rf.best_params_

