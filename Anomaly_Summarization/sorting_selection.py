# BASED ON THE CLASSIFICATION BASED ON THE SORTING METHOD, CREATES A SUMMARY
# CREATES AN OUTPUT VIDEO BY CONSIDERING A NEIGHBOURHOOD AROUND A CENTRAL FRAME

import numpy as np
import os
import statistics
import sys

from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from collections import OrderedDict


dataset = "bhopal_atm"
factor = 1
time_length = 9
neighbourhood = 40		# NUMBER OF FRAMES ON EITHER SIDE TO BE CONSIDERED FOR CREATION OF VIDEO
k = 0.2
video_root_path ='/media/hdd2/sukalyan/result'

# function to print the elements of a list (debugging)
def print_list(list_var):

	for i in list_var:
		print(i)


# function that loads the reconstruction error values from a file 
# it also returns the median and the standard deviation
def load_from_file():
	import os
	import sys

	# path where the output files are
	path = os.path.join(video_root_path,'{0}/result/{1}.txt'.format(dataset,sys.argv[1])) 

	f = open(path,"r")
	error_list = ([float((line.split('\n')[0]).split(',')[0]) for line in f])

	sorted_error_list = sorted(error_list)
		
	median_val = statistics.median(sorted_error_list)
	standard_deviation_val = statistics.pstdev(sorted_error_list)
	variance_val = statistics.pvariance(sorted_error_list)
	mean_val = statistics.mean(sorted_error_list)

	return median_val,standard_deviation_val,variance_val,mean_val


def create_movie():
	import cv2
	
	# ground_truth_path = os.path.join('{0}/groundtruth/{1}.txt'.format(dataset,sys.argv[1]))

	test_video_path = os.path.join(video_root_path,'{0}/testing_frames/jpg/{1}'.format(dataset,sys.argv[1]))

	count = 0
	output = sys.argv[1] + '_new.mp4'
	
	# only the 1st image has been saved
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

	# iterating over all the images (selecting only the required ones)
	for frame_folder in sorted(os.listdir(test_video_path)):
			
		image_path = os.path.join(test_video_path, frame_folder)
		frame = cv2.imread(image_path)
		out.write(frame) # Write out frame to video
		count = (count + 1 ) % factor



# function that assigns 0/1 to each frame and prints the f-measure by comparing the prediction with the ground truth
def classification_module(threshold):
	import sys
	import os
	import cv2

	# for prediction and calculating f-measure
	from sklearn.metrics import f1_score
	from sklearn.cluster import KMeans

	# for ROC curve
	from sklearn.metrics import roc_curve
	import matplotlib.pyplot as plt
	from sklearn.metrics import auc

	# location of the video files
	video_root_path ='/media/hdd2/sukalyan/input_videos'
	result_path = '/media/hdd2/sukalyan/result'

	# location of the groundtruth files
	ground_truth_path = os.path.join(video_root_path,'{0}/groundtruth/{1}.txt'.format(dataset,sys.argv[1]))
	result_path = os.path.join(result_path,'{0}/result/{1}.txt'.format(dataset,sys.argv[1]))

	output_val = []
	output_frame = []

	# loading the output data
	output_file = open(result_path,"r")

	# LOADING THE RECONSTRUCTION ERROR AND FRAME NUMBER INTO VARIABLES
	row = 0
	for line in output_file:
		x = line.split('\n')		
		x,y = x[0].split(',')
		output_val.append(float(x))
		output_frame.append(int(y))

		row = row + 1

	output_file.close()	


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
	result = [0] * row
	limit = round(k * row)
	
	# DECLARE AN ORDERED_MAP (HASH MAP) FOR KEEPING TRACK OF THE FRAMES THAT HAVE BEEN CONSIDERED FOR VIDEO GENERATION
	my_dict = dict() 

	for i in range(0,limit):				
		index = (i * factor) + 1

		# SETTING APPROPRIATE INDEX AS 1 (FOR ANOMALOUS FRAME ACCORDING TO THE CONDITION OF CLASSIFICATION)
		result[output_frame[i]-1] = 1
		
		# CHECKING IF THE PARENT FRAME HAS ALREADY BEEN USED FOR VIDEO CREATION
		if((output_frame[i]+1) in my_dict):
			continue

		# ITERATING FROM THE BEGINNING TOWARDS THE END (SUBTRACT MORE FIRST AND THEN REDUCE)
		for neigh in range(neighbourhood,0,-1):
			index_value = output_frame[i]+1-neigh
			if(index_value >=0 and index_value <=(row+time_length)):

				# VIDEO CREATION (THE LEFT HAND SIDE OF THE PARENT FRAME)
				image_loc = '{0}'.format(index_value) + '.jpg'
				image_path = os.path.join(test_video_path, image_loc )
				frame = cv2.imread(image_path)

				out.write(frame) # Write out frame to video
				
				# MARKING THE CURRENT INDEX AS VISITED
				my_dict[index_value] = 1
			
		# INCLUDING THE PARENT FRAME INTO THE VIDEO
		image_loc = '{0}'.format(output_frame[i]+1) + '.jpg'
		image_path = os.path.join(test_video_path, image_loc )
		frame = cv2.imread(image_path)
		out.write(frame) # Write out frame to video
		my_dict[output_frame[i]+1]=1

		for neigh in range(1,neighbourhood+1):
			index_value = output_frame[i]+1+neigh
			if(index_value >=0 and index_value <=(row+time_length)):

				# VIDEO CREATION (THE RIGHT HAND SIDE OF THE PARENT FRAME)
				image_loc = '{0}'.format(index_value) + '.jpg'
				image_path = os.path.join(test_video_path, image_loc )
				frame = cv2.imread(image_path)
				out.write(frame) # Write out frame to video
				my_dict[index_value] = 1
			
			
		
	# EVALUATION OF THE MODEL USING F-MEASURE		
	
	ground_truth_file = open(ground_truth_path,"r")
	
	# loading the ground truth values into a list
	ground_truth_val = list((int(line.split('\n')[0]) for line in ground_truth_file))
	ground_truth_file.close()

	true_positive = 0
	false_positive = 0
	true_negative = 0
	false_negative = 0

	# calculating true positive, true negative, false positives and false negatives
	print(row)
	for i in range(0,row):
    	
		if(ground_truth_val[i]==1):
			if(result[i]==ground_truth_val[i]):
				true_positive = true_positive + 1
			else:
				false_positive = false_positive + 1
		else:
			if(result[i]==ground_truth_val[i]):
				true_negative = true_negative + 1
			else:
				false_negative = false_negative + 1

	# ground_truth_val = ground_truth_val[:row]
	ground_truth_val = ground_truth_val[time_length : row + time_length]
	f_measure = f1_score(ground_truth_val, result, average='micro')

	fpr, tpr, thresholds = roc_curve(ground_truth_val, result)
	roc_auc = auc(fpr, tpr,reorder=True)
	
	# print(str(true_positive) + ',' + str(false_positive) + ',' + str(true_negative) + ',' + str(false_negative) + ',' + str(f_measure) + ',' + str(roc_auc))

	print("\n\n")
	print("True Positive: " + str(true_positive))
	print("False Positive: " + str(false_positive))
	print("True Negative: " + str(true_negative))
	print("False Negative: " + str(false_negative))
	print("F-Measure: " + str(f_measure))
	print("AUC of ROC: " + str(roc_auc))
	print("\n\n")

	
# MAIN PART OF THE PROGRAM (DRIVER)
median_value,standard_deviation_value,variance_value,mean_value = load_from_file()

print("Median value is: " + str(median_value))
print("Standard Deviation value is: " + str(standard_deviation_value))
print("Variance value is: "  + str(variance_value))
print("Mean value is: " + str(mean_value))


threshold = median_value - (standard_deviation_value)

print(sys.argv[1] + ".txt     Threshold: " + str(threshold) + "\n\n")

classification_module(threshold)







