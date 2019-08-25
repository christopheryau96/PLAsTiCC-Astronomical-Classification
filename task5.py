import pandas as pd
import numpy as np
import csv
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import MinimalFCParameters

#read training data
sub_pb = pd.read_csv('../modules/cs342/Assignment2/training_set.csv',header=0,usecols=[0,1,2,3],names=['object_id','mjd','passband','flux'])
sub_meta = pd.read_csv('../modules/cs342/Assignment2/training_set_metadata.csv',header=0,usecols=[0,11],names=['object_id','target'])

#create initial array for output prediction probabilities
rfc5outputfinal=np.zeros(16)
mlp5outputfinal=np.zeros(16)

#solve multi-processing problems on Windows
if __name__ == "__main__":
	
	#specify the features I want to extract. Specifically consider energy ratio in each of the 3 periods
	fc_parameters={
		'abs_energy': None,
		'cid_ce':[{'normalize':False}],
		'energy_ratio_by_chunks':[{'num_segments':3,'segment_focus':0},{'num_segments':3,'segment_focus':1},{'num_segments':3,'segment_focus':2}]}
		
	#data augmentation: consider each of the above features in different passbands
	kind_to_fc_parameters = {'0.0':fc_parameters,'1.0': fc_parameters,'2.0': fc_parameters,'3.0': fc_parameters,'4.0' : fc_parameters,'5.0': fc_parameters}

	train_features5=extract_features(sub_pb,column_id='object_id',column_value='flux',column_sort='mjd',column_kind='passband',
	default_fc_parameters=fc_parameters,kind_to_fc_parameters=kind_to_fc_parameters)
	impute(train_features5)

	#read test set in chunks to avoid memory issues
	for chunk in pd.read_csv('../modules/cs342/Assignment2/test_set.csv',header=0,usecols=[0,1,2,3],names=['object_id','mjd','passband','flux'],chunksize=10**6):
		sub_test_pb=chunk

#Task2 & Task3

		#extract training and test features using tsfresh
		test_features5=extract_features(sub_test_pb,column_id='object_id',column_value='flux',column_sort='mjd',column_kind='passband',
		default_fc_parameters=fc_parameters,kind_to_fc_parameters=kind_to_fc_parameters)	
		impute(test_features5)
		
#Task5

#RandomForestClassifier5
		training5_X = train_features5
		training5_Y = sub_meta['target']
		Task5_test = test_features5

		rfc5 = RandomForestClassifier(max_features='sqrt',n_estimators=500,min_samples_leaf=0.1,max_depth=4)
		rfc5_sigmoid = CalibratedClassifierCV(rfc5,cv=5,method='sigmoid')
		rfc5_sigmoid.fit(training5_X,training5_Y)
		rfc5output = rfc5_sigmoid.predict_proba(Task5_test)
		newcolumn=[0]*rfc5output.shape[0]
		for i in range(rfc5output.shape[0]):
			if sum(rfc5output[i,])>1:
				newcolumn[i] = 0
			else:
				newcolumn[i] = 1 - sum(rfc5output[i,])
		rfc5outputnew = np.c_[rfc5output,newcolumn]
		list = np.unique(np.unique(sub_test_pb['object_id']))
		rfc5outputnew = np.c_[list,rfc5outputnew]
		
		#stack the output to create a matrix of final output
		rfc5outputfinal = np.vstack((rfc5outputfinal,rfc5outputnew))

		
#MLPClassifier5
		mlp5 = MLPClassifier(alpha=1e-5,activation='logistic',solver='adam',learning_rate='constant',hidden_layer_sizes=(34))

		mlp5_sigmoid = CalibratedClassifierCV(mlp5,cv=5,method='sigmoid')
		mlp5_sigmoid.fit(training5_X,training5_Y)
		mlp5output = mlp5_sigmoid.predict_proba(Task5_test)
		newcolumn=[0]*mlp5output.shape[0]
		for i in range(mlp5output.shape[0]):
			if sum(mlp5output[i,])>1:
				newcolumn[i] = 0
			else:
				newcolumn[i] = 1 - sum(mlp5output[i,])
		mlp5outputnew = np.c_[mlp5output,newcolumn]
		list = np.unique(np.unique(sub_test_pb['object_id']))
		mlp5outputnew = np.c_[list,mlp5outputnew]
		
		#stack the output to create a matrix of final output
		mlp5outputfinal = np.vstack((mlp5outputfinal,mlp5outputnew))

#remove duplicate results (of the same object_id) due to reading the file in chunks		
	_,rfc5indices = np.unique(rfc5outputfinal[:,0],return_index=True)
	rfc5outputfinal2 = rfc5outputfinal[rfc5indices,:]
	_,mlp5indices = np.unique(mlp5outputfinal[:,0],return_index=True)
	mlp5outputfinal2 = mlp5outputfinal[mlp5indices,:]

#change dtype of object_id		
	rfc5outputfinal2=pd.DataFrame(rfc5outputfinal2,columns=['object_id','class_6','class_15','class_16','class_42','class_52','class_53','class_62','class_64','class_65','class_67','class_88','class_90','class_92','class_95','class_99'])
	rfc5outputfinal2=rfc5outputfinal2.astype({'object_id':int})
	
	mlp5outputfinal2=pd.DataFrame(mlp5outputfinal2,columns=['object_id','class_6','class_15','class_16','class_42','class_52','class_53','class_62','class_64','class_65','class_67','class_88','class_90','class_92','class_95','class_99'])
	mlp5outputfinal2=mlp5outputfinal2.astype({'object_id':int})
	
#output csv file for predictions		
rfc5outputfinal2.to_csv('rfc5.csv',index=False,float_format='%f',columns=['object_id','class_6','class_15','class_16','class_42','class_52','class_53','class_62','class_64','class_65','class_67','class_88','class_90','class_92','class_95','class_99'])
mlp5outputfinal2.to_csv('mlp5.csv',index=False,float_format='%f',columns=['object_id','class_6','class_15','class_16','class_42','class_52','class_53','class_62','class_64','class_65','class_67','class_88','class_90','class_92','class_95','class_99'])

