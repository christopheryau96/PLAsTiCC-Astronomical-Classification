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
rfc4outputfinal=np.zeros(16)
mlp4outputfinal=np.zeros(16)

#solve multi-processing problems on Windows
if __name__ == "__main__":
	
	#extract training and test features using tsfresh, here minimalfcparameters mean the basic statistic e.g. mean, max, min, sd etc.
	train_features4=extract_features(sub_pb,column_id='object_id',column_value='flux',column_sort='mjd',column_kind='passband',
	default_fc_parameters=MinimalFCParameters())
	impute(train_features4)
	
	#read test set in chunks to avoid memory issues
	for chunk in pd.read_csv('../modules/cs342/Assignment2/test_set.csv',header=0,usecols=[0,1,2,3],names=['object_id','mjd','passband','flux'],chunksize=10**6):
		sub_test_pb=chunk

#Task2 & Task3
		
		test_features4=extract_features(sub_test_pb,column_id='object_id',column_value='flux',column_sort='mjd',column_kind='passband',
		default_fc_parameters=MinimalFCParameters())	
		impute(test_features4)
	
#Task4

		training4_X = train_features4
		training4_Y = sub_meta['target']

		Task4_test = test_features4

#RandomForestClassifier4
		rfc4 = RandomForestClassifier(max_features='sqrt',n_estimators=400,min_samples_leaf=0.1,max_depth=5)
		
		#use calibratedclassifer such that calibrated probabilities for each class are predicted separately
		rfc4_sigmoid = CalibratedClassifierCV(rfc4,cv=5,method='sigmoid')
		rfc4_sigmoid.fit(training4_X,training4_Y)
		rfc4output = rfc4_sigmoid.predict_proba(Task4_test)
		newcolumn=[0]*rfc4output.shape[0]
		for i in range(rfc4output.shape[0]):
			if sum(rfc4output[i,])>1:
				newcolumn[i] = 0
			else:
				newcolumn[i] = 1 - sum(rfc4output[i,])
		rfc4outputnew = np.c_[rfc4output,newcolumn]
		list = np.unique(np.unique(sub_test_pb['object_id']))
		rfc4outputnew = np.c_[list,rfc4outputnew]
		
		#stack the output to create a matrix of final output
		rfc4outputfinal = np.vstack((rfc4outputfinal,rfc4outputnew))


#MLPClassifier4
		mlp4 = MLPClassifier(alpha=0.0001,activation='logistic',solver='adam',learning_rate='constant',hidden_layer_sizes=(38))

		#use calibratedclassifer such that calibrated probabilities for each class are predicted separately
		mlp4_sigmoid = CalibratedClassifierCV(mlp4,cv=5,method='sigmoid')
		mlp4_sigmoid.fit(training4_X,training4_Y)
		mlp4output = mlp4_sigmoid.predict_proba(Task4_test)
		newcolumn=[0]*mlp4output.shape[0]
		for i in range(mlp4output.shape[0]):
			if sum(mlp4output[i,])>1:
				newcolumn[i] = 0
			else:
				newcolumn[i] = 1 - sum(mlp4output[i,])
		mlp4outputnew = np.c_[mlp4output,newcolumn]
		list = np.unique(np.unique(sub_test_pb['object_id']))
		mlp4outputnew = np.c_[list,mlp4outputnew]
		
		#stack the output to create a matrix of final output
		mlp4outputfinal = np.vstack((mlp4outputfinal,mlp4outputnew))

#remove duplicate results (of the same object_id) due to reading the file in chunks		
	_,rfc4indices = np.unique(rfc4outputfinal[:,0],return_index=True)
	rfc4outputfinal2 = rfc4outputfinal[rfc4indices,:]
	_,mlp4indices = np.unique(mlp4outputfinal[:,0],return_index=True)
	mlp4outputfinal2 = mlp4outputfinal[mlp4indices,:]
	
#change dtype of object_id	
	rfc4outputfinal2=pd.DataFrame(rfc4outputfinal2,columns=['object_id','class_6','class_15','class_16','class_42','class_52','class_53','class_62','class_64','class_65','class_67','class_88','class_90','class_92','class_95','class_99'])
	rfc4outputfinal2=rfc4outputfinal2.astype({'object_id':int})
		
	mlp4outputfinal2=pd.DataFrame(mlp4outputfinal2,columns=['object_id','class_6','class_15','class_16','class_42','class_52','class_53','class_62','class_64','class_65','class_67','class_88','class_90','class_92','class_95','class_99'])
	mlp4outputfinal2=mlp4outputfinal2.astype({'object_id':int})
	
#output csv file for predictions			
rfc4outputfinal2.to_csv('rfc4.csv',index=False,float_format='%f',columns=['object_id','class_6','class_15','class_16','class_42','class_52','class_53','class_62','class_64','class_65','class_67','class_88','class_90','class_92','class_95','class_99'])
mlp4outputfinal2.to_csv('mlp4.csv',index=False,float_format='%f',columns=['object_id','class_6','class_15','class_16','class_42','class_52','class_53','class_62','class_64','class_65','class_67','class_88','class_90','class_92','class_95','class_99'])
