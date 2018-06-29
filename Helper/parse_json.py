# PROGRAM THAT READS A .json FILE GENERATED AS THE OUTPUT OF oTranscribe AND CREATES THE RATING FILES

import json
from pprint import pprint
import os
import datetime
import time


video_root_path = os.path.join('/media/hdd2/sukalyan/')
dataset = 'inhouse'

fps = 25
factor = 1
count = 0


with open('1.json') as f:
    data = json.load(f)

for i in range(0,len(data)):
	
	x = time.strptime(data[i]['begin'].split(',')[0],'%H:%M:%S')
	start = datetime.timedelta(hours=x.tm_hour,minutes=x.tm_min,seconds=x.tm_sec).total_seconds()
	

	x = time.strptime(data[i]['end'].split(',')[0],'%H:%M:%S')
	end = datetime.timedelta(hours=x.tm_hour,minutes=x.tm_min,seconds=x.tm_sec).total_seconds()

	rating = data[i]['rating']
	for j in range(int(start)*fps,(int(end)+1)*fps):
		frame = j * fps
		
		if (count % factor == 0):

			# CHANGE IF YOU WANT TO CREATE THE RATING FILE (SIMPLY PRINT THE VALUE FOR RATING, THIS CREATES THE GROUNDTRUTH)
			if(rating==-1):
				print(1)
			else:
				print(0)
		count = count + 1
		

		