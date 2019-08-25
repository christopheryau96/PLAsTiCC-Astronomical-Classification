import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

#read datasets
pb = pd.read_csv('../modules/cs342/Assignment2/training_set.csv',header=0)

#plot time series data for observation	
def pb_plots (objectid): 

	ob615 = pb.loc[pb['object_id'] == objectid]

	mjd0 = ob615.loc[ob615['passband']==0, 'mjd']
	flux0 = ob615.loc[ob615['passband']==0,'flux']
	mjd1 = ob615.loc[ob615['passband']==1,'mjd']
	flux1 = ob615.loc[ob615['passband']==1,'flux']
	mjd2 = ob615.loc[ob615['passband']==2,'mjd']
	flux2 = ob615.loc[ob615['passband']==2,'flux']
	mjd3 = ob615.loc[ob615['passband']==3,'mjd']
	flux3 = ob615.loc[ob615['passband']==3,'flux']
	mjd4 = ob615.loc[ob615['passband']==4,'mjd']
	flux4 = ob615.loc[ob615['passband']==4,'flux']
	mjd5 = ob615.loc[ob615['passband']==5,'mjd']
	flux5 = ob615.loc[ob615['passband']==5,'flux']

	f, axarr = plt.subplots(2,3)
	axarr[0,0].scatter(mjd0,flux0,color='red')
	axarr[0,0].set_title('passband_0')
	axarr[0,1].scatter(mjd1,flux1,color='blue')
	axarr[0,1].set_title('passband_1')
	axarr[0,2].scatter(mjd2,flux2,color='green')
	axarr[0,2].set_title('passband_2')
	axarr[1,0].scatter(mjd3,flux3,color='yellow')
	axarr[1,0].set_title('passband_3')
	axarr[1,1].scatter(mjd4,flux4,color='black')
	axarr[1,1].set_title('passband_4')
	axarr[1,2].scatter(mjd5,flux5,color='purple')
	axarr[1,2].set_title('passband_5')
	plt.suptitle(objectid)
	plt.show()

#plot one object_id from each of the 14 classes for observation
class_list = [2677,615,133773,1227,730,745,2922,3423,3910,4173,10757,713,18556]

for i in class_list:
	pb_plots(i)