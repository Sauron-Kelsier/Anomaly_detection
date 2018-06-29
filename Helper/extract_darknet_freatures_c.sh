# automating the feature extraction using Yolo
# saving the required things in the correct directory under each data set

#!/bin/bash


# store the videos in respective folder in '~/dataset/avenue/videos/training' 
dataset="inhouse"
train_test="training"
root_folder='/media/hdd2/sukalyan/yolo/yolo_9000/yolo-9000/darknet/'
layer=17
gpu=3
#penultimate_folder='media/hdd2/sukalyan/yolo/yolo9000-weights/yolo9000.weights'


# path name for the folders that will contain the .npy files
video_root_path="/media/hdd2/sukalyan/input_videos/"
video_folder=$video_root_path$dataset"/"$train_test"_frames/png"

# PATH NAME FOR FOLDERS THAT CONTAIN THE FRAMES
original_frame_training=$video_root_path$dataset"/training_frames/png"
original_frame_testing=$video_root_path$dataset"/testing_frames/png"
reconst_frame_training=$video_root_path$dataset"/reconstructed_training_frames"
reconst_frame_testing=$video_root_path$dataset"/reconstructed_testing_frames"

# two more will come (pathnames for the reconstructed frames ; training and testing)

# PATH NAME FOR THE FOLDERS THAT WILL CONTAIN THE FINAL .npy FILES
darknet_feature_testing_original=$video_root_path$dataset"/darknet_feature_testing_original"
darknet_feature_testing_reconstructed=$video_root_path$dataset"/darknet_feature_testing_reconst"
darknet_feature_training_original=$video_root_path$dataset"/darknet_feature_training_original"
darknet_feature_training_reconstructed=$video_root_path$dataset"/darknet_feature_training_reconst"

# PATH NAME FOR THE FOLDERS THAT WILL CONTAIN THE INTERMEDIATE INDIVIDUAL FRAME LEVEL .npy FILES THAT WILL BE COMBINED TO FORM THE BIGGER ONE
temp_original_training=$video_root_path$dataset"/original_training_temp"
temp_reconst_training=$video_root_path$dataset"/reconst_training_temp"
temp_original_testing=$video_root_path$dataset"/original_testing_temp"
temp_reconst_testing=$video_root_path$dataset"/reconst_testing_temp"



# checking if folder containing the merged .npy files are present or not (creating those folders)
if [ ! -d $darknet_feature_testing_original ];
then 
   mkdir $darknet_feature_testing_original
fi

if [ ! -d $darknet_feature_testing_reconstructed ];
then 
   mkdir $darknet_feature_testing_reconstructed
fi

if [ ! -d $darknet_feature_training_original ];
then 
   mkdir $darknet_feature_training_original
fi

if [ ! -d $darknet_feature_training_reconstructed ];
then 
   mkdir $darknet_feature_training_reconstructed
fi


# CREATING THE TEMPORARY DIRECTORIES

if [ ! -d $temp_original_training ];
then 
   mkdir $temp_original_training
fi

if [ ! -d $temp_original_testing ];
then 
   mkdir $temp_original_testing
fi

if [ ! -d $temp_reconst_training ];
then 
   mkdir $temp_reconst_training
fi

if [ ! -d $temp_reconst_testing ];
then 
   mkdir $temp_reconst_testing
fi





# EXTRACTING FEATURES FOR THE ORIGINAL TRAINING FRAMES (LAYER 17)
for d in $original_frame_training/*
do

   # CREATING A TEMPORARY DIRECTORY FOR EACH VIDEO FOR STORING INDIVIDUAL .npy FILES  
   # EXTRACTING THE FOLDER NAME
   directory=$(basename $d)
   dir_name=$temp_original_training"/"$directory


   echo $temp_original_training"/"$directory"/"$layer
   # TO BE DONE FOR EACH CASE

     
   # CREATING A DIRECTORY FOR A VIDEO AND A LAYER INSIDE THAT
   if [ ! -d $temp_original_training"/"$directory ];
   then
      mkdir $temp_original_training"/"$directory
      
      if [ ! -d $temp_original_training"/"$directory"/"$layer ];
      then 
         mkdir $temp_original_training"/"$directory"/"$layer
      fi
   fi
   
   
   # COMMENT IF YOU DO NOT WANT INDIVIDUAL .npy FILES TO BE GENERATED
   ./darknet detector test $root_folder"cfg/combine9k.data" $root_folder"cfg/yolo9000.cfg" $root_folder"../yolo9000-weights/yolo9000.weights" $original_frame_training"/"$directory -out $temp_original_training"/"$directory"/"$layer -i $gpu
    
         
   # creating the merged .npy files in the correct folders
   if [ ! -f $darknet_feature_training_original$name/".npy" ];
   then 
     
      # COMMENT IF YOU DO NOT WANT TO COMBINE THE INDIVIDUAL FILES
      python3 /media/hdd2/sukalyan/preprocess_after_darknet_MTP.py $temp_original_training"/" $directory $layer "training"
   fi
done


: '


# RECONSTRUCTED TRAINING FRAMES

# EXTRACTING FEATURES FOR RECONSTRUCTED TRAINING FRAMES
for d in $reconst_frame_training/*
do

   # CREATING A TEMPORARY DIRECTORY FOR EACH VIDEO FOR STORING INDIVIDUAL .npy FILES  
   # EXTRACTING THE FOLDER NAME
   directory=$(basename $d)
   dir_name=$temp_reconst_training"/"$directory


   echo $temp_reconst_training"/"$directory"/"$layer
   # TO BE DONE FOR EACH CASE

     
   # CREATING A DIRECTORY FOR A VIDEO AND A LAYER INSIDE THAT
   if [ ! -d $temp_reconst_training"/"$directory ];
   then
      mkdir $temp_reconst_training"/"$directory
      
      if [ ! -d $temp_reconst_training"/"$directory"/"$layer ];
      then 
         mkdir $temp_reconst_training"/"$directory"/"$layer
      fi
   fi
   
   # COMMENT IF YOU DO NOT WANT INDIVIDUAL .npy FILES TO BE GENERATED
    ./darknet detector test $root_folder"cfg/combine9k.data" $root_folder"cfg/yolo9000.cfg" $root_folder"../yolo9000-weights/yolo9000.weights" $reconst_frame_training"/"$directory -out $temp_reconst_training"/"$directory"/"$layer -i $gpu
    


   # REMOVE THIS IF CONDITION (IT IS NOT REQUIRED). THE PORTION THAT IS NOT REQUIRED WILL BE COMMENTED       
   # creating the merged .npy files in the correct folders
  
   if [ ! -f $darknet_feature_training_original$name/".npy" ];
   then 
   
      # COMMENT IF YOU DO NOT WANT TO COMBINE THE INDIVIDUAL FILES
      python3 /media/hdd2/sukalyan/preprocess_after_darknet_MTP.py $temp_reconst_training"/" $directory $layer $train_test
   fi
    
   # combine and create a new .npy file (DONE)
   
   # delete the temporary folder (THINK WHETHER REQUIRED OR NOT)
done
            


'






# EXTRACTING FEATURES FOR ORIGINAL TESTING FRAMES
for d in $original_frame_testing/*
do

   # CREATING A TEMPORARY DIRECTORY FOR EACH VIDEO FOR STORING INDIVIDUAL .npy FILES  
   # EXTRACTING THE FOLDER NAME
   directory=$(basename $d)
   dir_name=$temp_original_testing"/"$directory


   echo $temp_original_testing"/"$directory"/"$layer
   # TO BE DONE FOR EACH CASE

     
   # CREATING A DIRECTORY FOR A VIDEO AND A LAYER INSIDE THAT
   if [ ! -d $temp_original_testing"/"$directory ];
   then
      mkdir $temp_original_testing"/"$directory
      
      if [ ! -d $temp_original_testing"/"$directory"/"$layer ];
      then 
         mkdir $temp_original_testing"/"$directory"/"$layer
      fi
   fi
   
   # COMMENT IF YOU DO NOT WANT INDIVIDUAL .npy FILES TO BE GENERATED
    ./darknet detector test $root_folder"cfg/combine9k.data" $root_folder"cfg/yolo9000.cfg" $root_folder"../yolo9000-weights/yolo9000.weights" $original_frame_training"/"$directory -out $temp_original_testing"/"$directory"/"$layer -i $gpu
    


   # REMOVE THIS IF CONDITION (IT IS NOT REQUIRED). THE PORTION THAT IS NOT REQUIRED WILL BE COMMENTED       
   # creating the merged .npy files in the correct folders
  
   if [ ! -f $darknet_feature_training_original$name/".npy" ];
   then 
   
      # COMMENT IF YOU DO NOT WANT TO COMBINE THE INDIVIDUAL FILES
      python3 /media/hdd2/sukalyan/preprocess_after_darknet_MTP.py $temp_original_testing"/" $directory $layer "testing"
   fi
    
   # combine and create a new .npy file (DONE)
   
   # delete the temporary folder (THINK WHETHER REQUIRED OR NOT)
done



: '

# RECONSTRUCTED TESTING FRAMES

# EXTRACTING FEATURES FOR RECONSTRUCTED TRAINING FRAMES
for d in $reconst_frame_testing/*
do

   # CREATING A TEMPORARY DIRECTORY FOR EACH VIDEO FOR STORING INDIVIDUAL .npy FILES  
   # EXTRACTING THE FOLDER NAME
   directory=$(basename $d)
   dir_name=$temp_reconst_testing"/"$directory


   echo $temp_reconst_testing"/"$directory"/"$layer
   # TO BE DONE FOR EACH CASE

     
   # CREATING A DIRECTORY FOR A VIDEO AND A LAYER INSIDE THAT
   if [ ! -d $temp_reconst_testing"/"$directory ];
   then
      mkdir $temp_reconst_testing"/"$directory
      
      if [ ! -d $temp_reconst_testing"/"$directory"/"$layer ];
      then 
         mkdir $temp_reconst_testing"/"$directory"/"$layer
      fi
   fi
   
   # COMMENT IF YOU DO NOT WANT INDIVIDUAL .npy FILES TO BE GENERATED
    ./darknet detector test $root_folder"cfg/combine9k.data" $root_folder"cfg/yolo9000.cfg" $root_folder"../yolo9000-weights/yolo9000.weights" $reconst_frame_testing"/"$directory -out $temp_reconst_testing"/"$directory"/"$layer -i $gpu
    


   # REMOVE THIS IF CONDITION (IT IS NOT REQUIRED). THE PORTION THAT IS NOT REQUIRED WILL BE COMMENTED       
   # creating the merged .npy files in the correct folders
  
   if [ ! -f $darknet_feature_testing_original$name/".npy" ];
   then 
   
      # COMMENT IF YOU DO NOT WANT TO COMBINE THE INDIVIDUAL FILES
      python3 /media/hdd2/sukalyan/preprocess_after_darknet_MTP.py $temp_reconst_testing"/" $directory $layer $train_test
   fi
   
done

'

