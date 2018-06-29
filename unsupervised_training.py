# UNSUPERVISED AUTOENCODER TRAINING CODE

import tensorflow as tf 
import numpy as np
import h5py
import os
import random
import scipy.misc
import data_preprocessing as prep

from pathlib import Path
from tensorflow.contrib.keras import layers
from keras import backend as K
from tqdm import tqdm
from scipy.spatial.distance import euclidean


from keras.models import Model
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Activation
from keras.layers import Input


video_root_path ='/media/hdd2/sukalyan/input_videos'
time_length = 10
batch_size = 10
dataset = 'bhopal_atm'

# CLASS FOR THE MODEL. OBJECT WILL BE INSTANTIATED AND USED FOR THE PROCESSING
class Autoencoder:
	def __init__(self):
		print("Model object created")

	# MODULE THAT CONTAINS THE ARCHITECTURE OF THE MODEL AND PASSES THE INPUT THROUGH THE MODEL
	def build(self,inp):

		# ENCODER
		
		# 1ST CONVOLUTION LAYER
		self.conv1 = self.conv_layer(inp,11,4,128,'conv1')
		self.conv1 = self.batch_norm(self.conv1)
		self.conv1 = self.relu_oper(self.conv1)
		
		# 2ND CONVOLUTION LAYER
		self.conv2 = self.conv_layer(self.conv1,5,2,64,'conv2')
		self.conv2 = self.batch_norm(self.conv2)
		self.conv2 = self.relu_oper(self.conv2)
		
		# CONVOLUTION LSTM
		self.conv_lstm_1 = self.conv_lstm(self.conv2,64,3,'convlstm1')

		self.conv_lstm_2 = self.conv_lstm(self.conv_lstm_1,32,3,'convlstm2')

		self.conv_lstm_3 = self.conv_lstm(self.conv_lstm_2,64,3,'convlstm3')

		# 1ST DECONVOLUTION LAYER
		self.deconv1 = self.deconv_layer(self.conv_lstm_3,5,2,128,'deconv1')
		self.deconv1 = self.batch_norm(self.deconv1)
		self.deconv1 = self.relu_oper(self.deconv1)

		# RECONSTRUCTION OF THE ORIGINAL IMAGE
		self.decoded = self.deconv_layer(self.deconv1,11,4,1,'deconv2')
		

		print("Build model successful")

		

	# MODULE THAT CREATES A TIME_DISTRIBUTED 2D-CONVOLUTION LAYER
	def conv_layer(self,bottom,kernel_dim,stride,out_channel,name):
	       
        # DEFINING A CONVOLUTION FILTER
        conv = TimeDistributed(Conv2D(out_channel, kernel_size=(kernel_dim, kernel_dim), padding='same', strides=(stride, stride), name=name))(bottom)
		return conv


	# MODULE THAT CREATES A TIME_DISTRIBUTED 2D-DECONVOLUTION LAYER
	def deconv_layer(self,bottom,kernel_dim,stride,out_channel,name):
		
		deconv = TimeDistributed(Conv2DTranspose(out_channel, kernel_size=(kernel_dim, kernel_dim), padding='same', strides=(stride, stride), name=name))(bottom)

		return deconv

	# MODULE THAT CREATES A TIME DISTRIBUTED CONVOLUTION LSTM
	def conv_lstm(self,bottom,out_channel,kernel_dim,name):
		
		lstm = ConvLSTM2D(out_channel, kernel_size=(kernel_dim, kernel_dim), padding='same', return_sequences=True, name=name)(bottom)

		return lstm

	
	# MODULE THAT DOES A TIME_DISTRIBUTED BATCH_NORMALIZATION ON A GIVEN TENSOR 
	def batch_norm(self,bottom):
		norm = TimeDistributed(BatchNormalization())(bottom)
		return norm


	# MODULE THAT EVALUATES A TIME_DISTRIBUTED RELU OPERATION 
	def relu_oper(self,bottom):
		relu = TimeDistributed(Activation('relu'))(bottom)
		return relu




#################################################### TRAINING MODULE ####################################################
#########################################################################################################################

# FUNCTION THAT DOES THE TRAINING
def train_model():

	# VARIABLE THAT STORES THE PATH WHERE THE MODEL WILL BE SAVED
	path_to_model = os.path.join(video_root_path,'{0}/train_{0}_t{1}.ckpt'.format(dataset,time_length))
	training_frame_path = os.path.join(video_root_path, '{0}/training_frames/jpg/'.format(dataset))
	input_vector_path = os.path.join(video_root_path,'{0}/input_training_vector/'.format(dataset))
	training_input_file_path = os.path.join(video_root_path,'{0}/training_input_{1}.h5'.format(dataset,time_length))
	checkpoint_file_path = os.path.join(video_root_path,dataset)

	# PARAMETERS FOR TRAINING
	iterations = 50

	# PRE-STEPS. CREATING AND NORMALIZING INPUT
	
	# FUNCTION THAT CALCLULATES MEAN, SAVES IN A FILE AND CREATES .npy FILES FOR ALL THE FRAMES
	prep.normalize_input(dataset)

	# CALL FOR THE FUNCTION THAT CREATES INPUT AND SAVES THEM IN .h5 FILES
	prep.create_input(dataset,time_length)

	# DEFINING THE OPTIMIZER AND LOSS FUNCTION
	opt = tf.train.AdagradOptimizer(0.01)	
	
	# LOSS FUNCTION IS MEAN_SQUARE_ERROR 
	loss_function = tf.losses.mean_squared_error(frame_placeholder,vgg.decoded)

	# DEFINE THE OPTIMIZATION ALONG WITH THE LOSS FUNCTION
	training_operation = opt.minimize(loss=loss_function)

	# SETUP ALL THE VARIABLES
	run_ready = tf.global_variables_initializer()

	# ACTUALLY INITIALIZING ALL THE SESSION VARIABLES
	sess.run(run_ready)

	# CHECKING IF TRAIN FILE EXISTS OR NOT 
	my_file = Path(path_to_model)

	# LIST OF NAMES OF ALL THE INPUT FILES
	file_list = os.listdir(training_frame_path)

	for i in range(0,iterations):
		random.shuffle(file_list)
		print("Iteration Number: " + str(i+1))

		# FOR EVERY INPUT FILE (SHUFFLED ORDER)
		for frame_folder in file_list:

			frame_path = os.path.join(input_vector_path,frame_folder+'_{0}.h5'.format(time_length)) # creating the path 
			print(frame_path)

			# LOADING DATA FROM THE FILE INTO A VARIABLE
			with h5py.File(frame_path, 'r') as hf:
				data = hf[dataset][:]
			num_volume = data.shape[0]

			# IF THE MODEL ALREADY EXISTS, THEN SIMPLY LOAD THE TRAINED MODEL AND CONTINUE TRAINING
			if my_file.is_file():
				saver.restore(sess,tf.train.latest_checkpoint(checkpoint_file_path))

			limit = num_volume // batch_size
			print(num_volume)
			if(num_volume%batch_size != 0):
				limit = limit + 1


			for index in tqdm(range(0,limit)):
				start = index * batch_size
				end = min(start + batch_size ,num_volume )				
				line = data[start:end]

				# CONTINUING WITH THE TRAINING / ACTUAL OPERATION 
				sess.run(training_operation,feed_dict={frame_placeholder:line})
				
			# SAVING THE MODEL PARAMETERS
			saver_path = saver.save(sess,path_to_model)
		
	print("Training complete")



# MAIN FUNCTION OF THE PROGRAM

# COMMENT IT WHEN YOU WANT TO USE THE GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

# SOME PARAMETER THAT SHOULD BE SET BEFORE RUNNING THE MODEL
K.set_learning_phase(1)

# CREATING THE TENSORFLOW AND KERAS SESSIONS
sess = tf.Session()
K.set_session(sess)

# CREATING AN OBJECT FOR BUILDING THE MODEL
vgg = Autoencoder()

training_frame_path = os.path.join(video_root_path, '{0}/training_frames/'.format(dataset))
training_input_file_path = os.path.join(video_root_path,'{0}/training_input_{1}.h5'.format(dataset,time_length))

# PLACEHOLDER FOR INPUT VIDEOS
frame_placeholder = tf.placeholder('float',[None,time_length,224,224,1])

# BUILD FUNCTION THAT 
vgg.build(frame_placeholder)
saver = tf.train.Saver()
train_model()
