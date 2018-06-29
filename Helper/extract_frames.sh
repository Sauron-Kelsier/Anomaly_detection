# extract frames of a given dataset (both test and train) (jpg and png)
# if both not required, comment the appropriate portion

#!/bin/bash

dataset="bhopal_atm"
video_root_path="/media/hdd2/sukalyan/input_videos/"
training_video_folder=$video_root_path$dataset"/training_videos"
testing_video_folder=$video_root_path$dataset"/testing_videos"

training_frame_folder=$video_root_path$dataset"/training_frames"
testing_frame_folder=$video_root_path$dataset"/testing_frames"


# extracting the training video frames

jpg_training=$training_frame_folder"/jpg"
png_training=$training_frame_folder"/png"

mkdir $jpg_training
mkdir $png_training


for f in $training_video_folder/*
do
   file_name=${f##*/}
   name=$(echo $file_name | cut -f 1 -d '.')

   new_jpg_dir=$jpg_training"/"$name
   new_png_dir=$png_training"/"$name
   
   mkdir $new_jpg_dir
   mkdir $new_png_dir
   ffmpeg -i $f -s 224x224 $new_jpg_dir/%d.jpg 
   ffmpeg -i $f -s 224x224 $new_png_dir/%d.png 
done


# extracting the testing video frames

jpg_testing=$testing_frame_folder"/jpg"
png_testing=$testing_frame_folder"/png"

mkdir $jpg_testing
mkdir $png_testing

for f in $testing_video_folder/*
do
   file_name=${f##*/}
   name=$(echo $file_name | cut -f 1 -d '.')

   new_jpg_dir=$jpg_testing"/"$name
   new_png_dir=$png_testing"/"$name
   
   mkdir $new_jpg_dir
   mkdir $new_png_dir
   
   ffmpeg -i $f -s 224x224 $new_jpg_dir/%d.jpg  
   ffmpeg -i $f -s 224x224 $new_png_dir/%d.png
done




