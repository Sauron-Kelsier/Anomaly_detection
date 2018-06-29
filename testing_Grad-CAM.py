# PROGRAM THAT PRINTS THE RECONSTRUCTION ERROR (TESTING)
# DOES IT BATCH WISE
# SEMI-SUPERVISED TESTING
# SAVES THE GRAD-CAM IMAGES IN THE CORRECT FOLDER
# CONTAINS CODE THAT SORTS THE FRAMES IN THE DECREASING ORDER OF RECONSTRUCTION ERROR

import tensorflow as tf 
import numpy as np
import h5py
import os
import random
import scipy.misc
import data_preprocessing as prep
from operator import itemgetter


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
dataset = 'inhouse'



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




	# SOMETHING CALLED 'return_sequences = True'. ALSO, PADDING ISN'T PRESENT.
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


#################################################### TESTING MODULE #####################################################
#########################################################################################################################

def test_model():

	# VARIABLE THAT STORES THE PATH WHERE THE MODEL WILL BE SAVED
	path_to_model = os.path.join(video_root_path,'{0}/train_{0}_t{1}.ckpt'.format(dataset,time_length))

	testing_frame_path = os.path.join(video_root_path, '{0}/testing_frames/jpg/'.format(dataset))
	input_vector_path = os.path.join(video_root_path,'{0}/input_testing_vector/'.format(dataset))
	training_input_file_path = os.path.join(video_root_path,'{0}/testing_input_{1}.h5'.format(dataset,time_length))
	output_file_path = os.path.join(video_root_path,dataset)

	# LOADING THE GRAPH AND THE PARAMETERS
	meta_file_path = os.path.join(video_root_path,dataset,'train_avenue_t10.ckpt.meta')
	checkpoint_file_path = os.path.join(video_root_path,dataset)
	saver.restore(sess,tf.train.latest_checkpoint(checkpoint_file_path))
	
	# CREATING A GRAPH FROM THE SAVED GRAPH
	graph = tf.get_default_graph()

	file_list = sorted([os.path.join(input_vector_path, item) for item in os.listdir(input_vector_path)])

	
	# Grad-CAM STEPS 
	y = vgg.decoded
	x = vgg.deconv1

	# CALCULATING THE GRADIENT
	grad = tf.gradients(y,x)
	
	# REDUCTION ALONG ALL THE AXES EXCEPT THE NUMBER OF FRAMES AND OUT_CHANNELS (REMOVE 0 IF YOU ARE USING A BATCH)
	Z = tf.reduce_sum(grad,axis=[4,3,2,1])

	# POINT-WISE MULTIPLICATION OF CONVOLUTION LAYER WITH THE IMPORTANCE SCORE
	rel = tf.multiply(vgg.deconv1,Z)

	# REDUCTION ALONG THE OUTPUT CHANNELS
	rel = tf.reduce_sum(rel,axis = 4)

	# PASSING THE REDUCED CONVOLUTION FILTER THROUGH ReLU IN ORDER TO CLIP THE NEGATIVE VALUES
	rel = tf.nn.relu(rel)

	for n, f in enumerate(file_list):
		with h5py.File(f, 'r') as hf:
			test_data = hf[dataset][:]
			num_volume = test_data.shape[0]
			print(f)
			limit = num_volume // batch_size
			if(num_volume%batch_size != 0):
				limit = limit + 1

			count = 0

			# ARRAY THAT WILL STORE THE FRAME NUMBER ALONG WITH THE RECONSTRCUCTION ERROR VALUE
			to_be_sorted = []

			# EXTRACT FOLDERNAME 
			filename_w_ext = os.path.basename(f)
			grad_file, file_extension = os.path.splitext(filename_w_ext)
			grad_file = grad_file.split('_')[0]
			grad_folder = os.path.join(video_root_path,dataset,'grad_cam',str(int(grad_file)))
			print(grad_folder)

			# CHECKING IF FOLDER UNDER GRAD_CAM DIRECTORY EXISTS OR NOT, IF NOT CREATE 
			if not os.path.exists(grad_folder):
				os.makedirs(grad_folder)

			for index in range(0,limit):
				start = index * batch_size
				end = min(start + batch_size ,num_volume )
				
				vec = test_data[start:end]		
				
				# OUTPUT OF THE FINAL LAYER OF THE NETWORK
				x = sess.run(y,feed_dict={frame_placeholder:vec})
				grad_x = sess.run(rel,feed_dict={frame_placeholder:vec})
				
				last = min(batch_size,x.shape[0])
				for i in range(0,last):
					
					# CONSIDERING THE ENTIRE VOLUME FOR THE DISTANCE MEASURE
					arg1 = vec[i].flatten()
					arg2 = x[i].flatten()
					ans = euclidean(arg1,arg2)
					count = count + 1
					file_name = os.path.join(grad_folder,'{0}.png'.format(count))

					# BILINEAR INTERPOLATION STEP OF THE Grad-CAM PROCEDURE				
					img = scipy.misc.imresize(grad_x[0][i],(224,224),interp='bilinear')
					scipy.misc.imsave(file_name, img)

					# SORTED ARRAY
					to_be_sorted.append([ans,count])				

			# SORTING IN DESCENDING ORDER
			to_be_sorted = sorted(to_be_sorted,key=itemgetter(0),reverse=True)
			
			# FOR WRITING INTO THE FILE
			for it in to_be_sorted:
				print(str(it[0]) + ',' + str(it[1]))
			
			print("\n\n")					



# COMMENT IT WHEN YOU WANT TO USE THE GPU
os.environ['CUDA_VISIBLE_DEVICES'] = ''

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

# BUILD FUNCTION 
vgg.build(frame_placeholder)
saver = tf.train.Saver()
test_model()


