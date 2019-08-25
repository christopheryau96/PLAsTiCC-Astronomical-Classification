import pandas as pd
import numpy as np
import csv
import random
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.utils import to_categorical

#read training data
sub_pb = pd.read_csv('../modules/cs342/Assignment2/training_set.csv',header=0,usecols=[0,1,2,3],names=['object_id','mjd','passband','flux'])
sub_meta = pd.read_csv('../modules/cs342/Assignment2/training_set_metadata.csv',header=0,usecols=[0,11],names=['object_id','target'])

#create initial array for output prediction probabilities
cnn6outputfinal=np.zeros(16)

#create training data
training6_X = np.zeros((sub_meta.shape[0],256))
count = 0
for i in np.unique(sub_pb['object_id']):

	object = sub_pb[sub_pb['object_id']==i]
	flux = object['flux']
	if len(flux)>256:
		training6_X[count] = random.sample(flux,256)
	else:
		extralength = 256 - len(flux)
		vector = np.r_[flux,[0]*extralength]
		training6_X[count] = vector
	count = count +1

training6_X = training6_X.reshape(-1,256,1)

#reset the classes for the to_categorical function 
target=sub_meta['target']
target[target==6] = 0
target[target==15] = 1
target[target==16] = 2
target[target==42] = 3
target[target==52] = 4
target[target==53] = 5
target[target==62] = 6
target[target==64] = 7
target[target==65] = 8
target[target==67] = 9
target[target==88] = 10
target[target==90] = 11
target[target==92] = 12
target[target==95] = 13

#create the desired training output format for keras
training6_Y = to_categorical(target,15)


#create cnn
model = Sequential()
model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(256,1)))
model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(15, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(training6_X, training6_Y, epochs=10, batch_size=32, verbose=0)

#read test set in chunks to avoid memory issues	
for chunk in pd.read_csv('../modules/cs342/Assignment2/test_set.csv',header=0,usecols=[0,1,2,3],names=['object_id','mjd','passband','flux'],chunksize=10**6):
	sub_test_pb=chunk
	
	#create test data
	task6_test = np.zeros((len(np.unique(sub_test_pb['object_id'])),256))
	count = 0
	for i in np.unique(sub_test_pb['object_id']):

		object = sub_test_pb[sub_test_pb['object_id']==i]
		flux = object['flux']
		if len(flux)>256:
			task6_test[count] = random.sample(flux,256)
		else:
			extralength = 256 - len(flux)
			vector = np.r_[flux,[0]*extralength]
			task6_test[count] = vector
		count = count +1

	task6_test = task6_test.reshape(-1,256,1)
	
	#predict
	cnn6output = model.predict(task6_test)
	list = np.unique(sub_test_pb['object_id'])
	cnn6output = np.c_[list,cnn6output]
	
	#stack the output to create a matrix of final output
	cnn6outputfinal = np.vstack((cnn6outputfinal,cnn6output))


#remove duplicate results (of the same object_id) due to reading the file in chunks
_,cnn6indices = np.unique(cnn6outputfinal[:,0],return_index=True)
cnn6outputfinal2 = cnn6outputfinal[cnn6indices,:]

#change dtype of object_id	
cnn6outputfinal2=pd.DataFrame(cnn6outputfinal2,columns=['object_id','class_6','class_15','class_16','class_42','class_52','class_53','class_62','class_64','class_65','class_67','class_88','class_90','class_92','class_95','class_99'])
cnn6outputfinal2=cnn6outputfinal2.astype({'object_id':int})

#output csv file for predictions
cnn6outputfinal2.to_csv('cnn6.csv',index=False,float_format='%f',columns=['object_id','class_6','class_15','class_16','class_42','class_52','class_53','class_62','class_64','class_65','class_67','class_88','class_90','class_92','class_95','class_99'])

