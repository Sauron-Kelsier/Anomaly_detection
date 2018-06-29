#!/bin/bash

max=$1
echo "Filename,Median,Std_Dev,Threshold,True_Pos,False_Pos,True_Neg,False_Neg,F1_Score,AUC"
for i in `seq 1 $max`
do
       python3 sliding_window.py $i
#     python3 sorting_selection.py $i
#      python3 thresholding.py $i

done
