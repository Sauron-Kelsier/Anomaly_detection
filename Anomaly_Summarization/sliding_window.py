# PROGRAM THAT MAINTAINS A SLIDING WINDOW AND FINDS THE INTERVAL WITH THE MAXIMUM WEIGHT

import sys
import os
import cv2

# location of the video files
video_root_path ='/media/hdd2/sukalyan/input_videos'
dataset = "avenue"

# WINDOW SIZE DENOTES THE NUMBER OF FRAMES THAT WILL CONSTITUTE THE CLIQUE
window_size = 200
factor = 1


def classification():

	# READING THE OUTPUT FILES AND POPULATING THE REQUIRED DATA STRUCTURES THAT WILL BE USED FOR PROCESSING

	# LOCATION WHERE THE RESULT FILES WILL BE STORED
	result_path = '/media/hdd2/sukalyan/result'

	# LOCATION OF THE GROUNDTRUTH FILES (FOR EVALUATION PURPOSE)
	ground_truth_path = os.path.join(video_root_path,'{0}/groundtruth/{1}.txt'.format(dataset,sys.argv[1]))
	result_path = os.path.join(result_path,'{0}/result/{1}.txt'.format(dataset,sys.argv[1]))

	# DATA STRUCTURES THAT STORE THE RECONSTRUCTION ERROR AND THE FRAME NUMBERS
	output_val = []
	output_frame = []
	
	# loading the output data
	output_file = open(result_path,"r")

	# LOADING THE RECONSTRUCTION ERROR AND FRAME NUMBER INTO VARIABLES
	row = 0
	for line in output_file:
		x = line.split('\n')		
		output_val.append(float(x[0]))
		row = row + 1

	output_file.close()	

	# SLIDING THE WINDOW IN ORDER TO FIND THE MAXIMUM VALUE 
	# INITIALIZING ans THAT WILL CONTAIN THE ANSWER
	ans = 0

	# IN CASE THE LENGTH OF THE VIDEO IS LESSER THAN THE VALUE OF WINDOW 
	limit = min(window_size,row)

	# ADDING THE RECONSTRUCTION ERROR OF THE FIRST WINDOW
	s = 0
	for i in range(0,limit):
		s = s + output_val[i]

	# INITIALIZING ans TO THE SUM OF THE FIRST WINDOW
	ans = s
	begin = 0

	# CONTINUING WITH THE REST OF THE INPUT
	for i in range(1,row-limit):
		s = s - output_val[i] + output_val[i+limit]
		if(s > ans):
			begin = i
			ans = s

	

######################################################################################################################
	# setting up for video creation

	test_video_path = os.path.join(video_root_path,'{0}/testing_frames/jpg/{1}'.format(dataset,sys.argv[1]))

	count = 0
	output = sys.argv[1] + '_new.avi'

	images = []
	for f in os.listdir(test_video_path):
	    images.append(f)
	    break

	# Determine the width and height from the first image
	image_path = os.path.join(test_video_path, images[0])
	frame = cv2.imread(image_path)
	height, width, channels = frame.shape
	fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
	out = cv2.VideoWriter(output, fourcc, 20.0, (width, height))

######################################################################################################################	

	# list that will store the final classification
	for i in range(begin,begin + window_size):	
		index = (i * factor) + 1
		image_loc = '{0}'.format(i+1) + '.jpg'
		image_path = os.path.join(test_video_path, image_loc )
		frame = cv2.imread(image_path)
		out.write(frame) # Write out frame to video

classification()