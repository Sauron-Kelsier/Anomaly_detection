# PROGRAM THAT GENERATES RATING FILE FOR TRAINING DATA (NORMAL EVENTS ONLY)

import tensorflow as tf 
import numpy as np
import h5py
import os
import random
import scipy.misc
from skimage.io import imread
import glob   


from pathlib import Path


from tqdm import tqdm


video_root_path ='/media/hdd2/sukalyan/input_videos'
time_length = 10
factor = 4
dataset = 'inhouse'


# FUNCTION THAT CREATES .txt FILES THAT CONTAIN THE FRAME-WISE RATING FOR EACH TRAINING_VIDEO
def rate_train():
	
	training_frame_path = os.path.join(video_root_path, '{0}/training_frames/jpg/'.format(dataset))
	input_vector_path = os.path.join(video_root_path,'{0}/input_training_vector/'.format(dataset))
	video_vector_path = os.path.join(video_root_path,'{0}/training_video_vector'.format(dataset))
	rating_path = os.path.join(video_root_path,'{0}/rating_training'.format(dataset))


	# create a directory for saving the normalized .npy files for each video
	os.makedirs(rating_path,exist_ok=True)

	for frame_folder in sorted(os.listdir(training_frame_path)):
		frame_path = os.path.join(training_frame_path,frame_folder)		

		# if input file not present, go ahead and create the input file
		images=glob.glob(frame_path+"/*.jpg")
		rating_file = str(int(frame_folder))
		rating_file_path = os.path.join(rating_path,'{0}.txt'.format(rating_file))	
		output_file = open(rating_file_path, "w")

		dim = np.shape(images)[0]

		for i in range(0,dim):
			if(i%factor == 0):
				output_file.write(str(1)+"\n")

		output_file.close()		

rate_train()