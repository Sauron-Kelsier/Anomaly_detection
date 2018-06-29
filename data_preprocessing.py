# DATA-PREPROCESSING THAT IS USED BY THE INITIAL PART OF THE TRAINING PROCESS

import tensorflow as tf 
import numpy as np
import h5py
import os
import random
import scipy.misc
from skimage.io import imread
import glob   


from pathlib import Path
from tensorflow.contrib.keras import layers
from keras import backend as K
from tqdm import tqdm


from keras.models import Model
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Activation
from keras.layers import Input

video_root_path ='/media/hdd2/sukalyan/input_videos'
time_length = 10

# function that calculates the mean frame value
# function that stores every video frame as .npy file that can be later loaded for creating volumes in 'create_input'
def normalize_input(dataset):

	training_frame_path = os.path.join(video_root_path, '{0}/training_frames/jpg/'.format(dataset))
	input_vector_path = os.path.join(video_root_path,'{0}/input_training_vector/'.format(dataset))
	video_vector_path = os.path.join(video_root_path,'{0}/training_video_vector'.format(dataset))

	mean_file_path = os.path.join(video_root_path,'{0}/mean_frame.npy'.format(dataset))
	print(mean_file_path)

########################################################################################################################
# calculating the mean value of all the frames of a video and storing them (this is part of normalization)

	
	# checking if mean value file is present or not
	if not os.path.isfile(mean_file_path):

		mean_frame = np.zeros((224, 224)).astype('float16')	# the frame that will store the mean value

		count = 0			# count for calculating the average
		# iterating over all the frames
		for frame_folder in sorted(os.listdir(training_frame_path)):
			frame_path = os.path.join(training_frame_path,frame_folder)

			images=glob.glob(frame_path+"/*.jpg")
				
			# reading the images in the correct order
			# creating the individual h5 files (extracting image vectors for every image)
			for f in sorted(images):
				
			    name = f.split('.')
			    name = name[0].split('/')
			    
			    # reading the image as greyscale
			    img = imread(f, as_grey=True)

			    assert(0. <= img.all() <= 1.)
			     count = count + 1
			    mean_frame = mean_frame + img
			    
		# calculating the average value for all the frames 
		mean_frame = mean_frame / count
		
		# saving the mean frame value in an .npy file
		np.save(mean_file_path,mean_frame)

		print("Mean file created")
			
	else:
 		print("Mean file already present. If you want to redo the operation, please delete the file")


########################################################################################################################
# subtracting mean value from all the frames of the video. This is prior to creating volumes that will be fed as input

	# loading the mean frame from file 
	mean_frame = np.load(mean_file_path)
	factor = 4

	# creating .npy files for the TRAINING DATA FILES
	print("TRAINING .npy FILES BEING CREATED")

	# create a directory for saving the normalized .npy files for each video
	os.makedirs(video_vector_path,exist_ok=True)

	video_vector_path = os.path.join(video_root_path,'{0}/training_video_vector/'.format(dataset))

	for frame_folder in sorted(os.listdir(training_frame_path)):
		frame_path = os.path.join(training_frame_path,frame_folder)

		individual_video_path =  os.path.join(video_root_path,'{0}/training_video_vector/{1}.npy'.format(dataset,frame_folder))

		# if input file not present, go ahead and create the input file
		if not os.path.isfile(individual_video_path):
		
			images=glob.glob(frame_path+"/*.jpg")

			image_vector = []
			count = 0

			for f in sorted(images):
				name = f.split('.')
				name = name[0].split('/')
				    
				# reading the image as greyscale
				img = imread(f, as_grey=True)

				assert(0. <= img.all() <= 1.)

				# selecting only few frames and not all the 24 frames per second (depends on the factor)
				if((count % factor) == 0):
					img = img - mean_frame
					image_vector.append(img)
				count = count + 1

			# creating the image vector
			image_vector = np.array(image_vector)
			print(np.shape(image_vector))
			image_vector = np.expand_dims(image_vector,axis=-1)
			print(np.shape(image_vector))

			# save the .npy files for each video in the location (this will be used later for creating volumes)
			file_path = os.path.join(individual_video_path)
			np.save(file_path,image_vector)
			print("Training: "+str(frame_folder)+".npy created")

		# write the message that the file is already present
		else:
			print("Training: "+str(frame_folder)+".npy already present")
		


	print("TESTING .npy FILES BEING CREATED")

	# creating .npy files for the TESTING DATA FILES

	# create a directory for saving the normalized .npy files for each video
	training_frame_path = os.path.join(video_root_path, '{0}/testing_frames/jpg/'.format(dataset))
	input_vector_path = os.path.join(video_root_path,'{0}/input_testing_vector/'.format(dataset))
	# video_vector_path = os.path.join(video_root_path,'{0}/testing_video_vector'.format(dataset))

	video_vector_path = os.path.join(video_root_path,'{0}/testing_video_vector/'.format(dataset))
		
	os.makedirs(video_vector_path,exist_ok=True)	

	for frame_folder in sorted(os.listdir(training_frame_path)):
		frame_path = os.path.join(training_frame_path,frame_folder)

		individual_video_path =  os.path.join(video_root_path,'{0}/testing_video_vector/{1}.npy'.format(dataset,frame_folder))

		# if input file not present, go ahead and create the input file
		if not os.path.isfile(individual_video_path):
		
			images=glob.glob(frame_path+"/*.jpg")

			image_vector = []
			count = 0

			for f in sorted(images):
				name = f.split('.')
				name = name[0].split('/')
				    
				# reading the image as greyscale
				img = imread(f, as_grey=True)

				assert(0. <= img.all() <= 1.)

				# selecting only few frames and not all the 24 frames per second (depends on the factor)
				if((count % factor) == 0):
					img = img - mean_frame
					image_vector.append(img)
				count = count + 1

			# creating the image vector
			image_vector = np.array(image_vector)
			print(np.shape(image_vector))
			image_vector = np.expand_dims(image_vector,axis=-1)
			print(np.shape(image_vector))

			# save the .npy files for each video in the location (this will be used later for creating volumes)
			file_path = os.path.join(individual_video_path)
			np.save(file_path,image_vector)
			print(str(frame_folder)+".npy created")

		# write the message that the file is already present
		else:
			print(str(frame_folder)+".npy already present")

	print("Done")
	return




# function that loads pixel values into np array for representing frames
def create_input(dataset,time_length):
	import os
	from pathlib import Path
	from keras.models import load_model
	from keras.callbacks import ModelCheckpoint
	from tqdm import tqdm

	# TRAINING VOLUMES BEING CREATED

	print("TRAINING .h5 FILES BEING CREATED")

	# contains the frames of training videos
	training_frame_path = os.path.join(video_root_path, '{0}/training_frames/jpg/'.format(dataset))
	
	# location where the volume .h5 files are stored that are used as input for training the model
	input_vector_path = os.path.join(video_root_path,'{0}/input_training_vector/'.format(dataset))

	# location where the .npy files for each video is saved. For volume creation, these files are extracted and used
	video_vector_path = os.path.join(video_root_path,'{0}/training_video_vector'.format(dataset))

	os.makedirs(input_vector_path,exist_ok=True)

	data = []
	
	# listing every frame from the training directory in the correct order
	for frame_folder in sorted(os.listdir(training_frame_path)):
		frame_path = os.path.join(training_frame_path,frame_folder)

		# location for each video .npy file
		individual_video_path =  os.path.join(video_root_path,'{0}/training_video_vector/{1}.npy'.format(dataset,frame_folder))

		# creating the address of the h5 file for storing the input matrix
		# h5 file name is the same as the name of the video
		file_path = os.path.join(video_root_path,'{0}/input_training_vector/{1}_{2}.h5'.format(dataset,frame_folder,time_length))
		h5_file_name = os.path.join('{0}_{1}'.format(frame_folder,time_length))

		# check if the volume file already exists or not, if it does, then continue
		if(os.path.isfile(file_path)):
			print(h5_file_name+".h5 already exists")
			continue
		
		image_vector = np.load(individual_video_path)

		# creating volumes of frames for every individual video
		num_frame = image_vector.shape[0]
		print(num_frame)
		data_only_frame = np.zeros((num_frame-time_length+1, time_length, 224, 224, 1)).astype('float16')

		vol = 0
		for j in range(num_frame-time_length+1):
			data_only_frame[vol] = image_vector[j:j+time_length] # Read a single volume
			vol += 1

		data = np.array(data_only_frame)
			
		# creating the input file as a .h5 file
		with h5py.File(file_path, 'w') as hf:
			hf.create_dataset(dataset,  data=data)

		print(h5_file_name+ " created")



	# TESTING VOLUMES BEING CREATED
	print("TESTING .h5 FILES BEING CREATED")

	# contains the frames of training videos
	training_frame_path = os.path.join(video_root_path, '{0}/testing_frames/jpg/'.format(dataset))
	
	# location where the volume .h5 files are stored that are used as input for training the model
	input_vector_path = os.path.join(video_root_path,'{0}/input_testing_vector/'.format(dataset))

	# location where the .npy files for each video is saved. For volume creation, these files are extracted and used
	video_vector_path = os.path.join(video_root_path,'{0}/testing_video_vector'.format(dataset))

	os.makedirs(input_vector_path,exist_ok=True)

	data = []
	
	# listing every frame from the training directory in the correct order
	for frame_folder in sorted(os.listdir(training_frame_path)):
		frame_path = os.path.join(training_frame_path,frame_folder)

		# location for each video .npy file
		individual_video_path =  os.path.join(video_root_path,'{0}/testing_video_vector/{1}.npy'.format(dataset,frame_folder))

		# creating the address of the h5 file for storing the input matrix
		# h5 file name is the same as the name of the video
		file_path = os.path.join(video_root_path,'{0}/input_testing_vector/{1}_{2}.h5'.format(dataset,frame_folder,time_length))
		h5_file_name = os.path.join('{0}_{1}'.format(frame_folder,time_length))

		# check if the volume file already exists or not, if it does, then continue
		if(os.path.isfile(file_path)):
			print(h5_file_name+".h5 already exists")
			continue		

		image_vector = np.load(individual_video_path)
		
		# creating volumes of frames for every individual video
		num_frame = image_vector.shape[0]
		print(num_frame)
		data_only_frame = np.zeros((num_frame-time_length+1, time_length, 224, 224, 1)).astype('float16')

		vol = 0
		for j in range(num_frame-time_length+1):
			data_only_frame[vol] = image_vector[j:j+time_length] # Read a single volume
			vol += 1

		data = np.array(data_only_frame)
			
		# creating the input file as a .h5 file
		with h5py.File(file_path, 'w') as hf:
			hf.create_dataset(dataset,  data=data)

		print(h5_file_name+ " created")

#########################################################################################################################
################################################ END OF INPUT CREATION ##################################################

