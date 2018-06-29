# PROGRAM THAT COMBINES FEATURES EXRACTED USING YOLO (SAVED IN A FILE AND READS THEM TO CREATE A MERGED .npy FILE)
# DONE FOR THE ACTUAL DATASETS (USE THIS)

# FOR EXECUTING THE PROGRAM
# python3 preprocess_after_darknet.py <input_directory> <dataset> <layer_number> <traing_test>

import tensorflow as tf 
import numpy as np
import h5py
import os
import random
import scipy.misc
import sys
from skimage.io import imread
import glob   

# LOCATION OF THE FEATURES
feature_file_path_name = "/media/hdd2/sukalyan/yolo/darknet/dataset/"

# ROOT LOCATION WHERE THE MERGED FEATURES WILL BE STORED
video_root_path ='/media/hdd2/sukalyan/input_videos/'

# DATASET ON WHICH THE MODEL IS WORKING ON 
dataset = "bhopal_atm"
train_test = "training"
orig_reconst = "original"

# FEATURES EXTRACTED FROM WHICH LAYER NUMBER
layer = "17"


# FUNCTION THAT READS THE FEATURES AND CREATES NP ARRAYS
# OUTPUT OF THIS FUNCTION WILL BE PICKED UP FROM THE EXISTING 'data_preprocessing.py' FILE
def read_dat_file(filepath,filename,layer=layer,train_test=train_test):

			
	# PATHNAME WHERE THE SOURCE .npy FILES ARE (FILE THAT HAVE TO BE COMBINED)
	final_path_name = os.path.join(filepath,filename,layer)
	print(final_path_name)
	
	dat_file = glob.glob(final_path_name+"/*.dat")

	# CHECKING THE NUMBER OF FILES
	num_files = len(dat_file)	
	

	# CREATING AN ARRAY THAT CONTAINS THE COMBINED INFORMATION
	feature_array = []
	overall_feature = []
	for i in range(1,num_files+1):
		
		# CREATING THE PATHNAME FOR EACH FEATURE FILE
		temp = final_path_name + '/' + str(layer) + '_' + str(i) + '.dat'
		feature_array = np.fromfile(temp,dtype=float)

		overall_feature.append(feature_array)
				
	
	# CONVERTING THE MERGED FEATURE VECTOR INTO A FILE
	overall_feature = np.array(overall_feature)
	merged_feature_vector_file = os.path.join(video_root_path,'{0}/darknet_feature_{1}_{2}/'.format(dataset,train_test,orig_reconst))

	
	saving_file_name = merged_feature_vector_file +  str(filename) + '.npy'
	print(saving_file_name)

	np.save(saving_file_name,overall_feature)

	
# 1st param - filepath
# 2nd param - filename
# 3rd param - layer number (has a default value)
# 4th param - training / testing
read_dat_file(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])


# FOR EXECUTION
# python3 preprocess_after_darknet_MTP.py input_videos/avenue/original_training_temp 01 17 training