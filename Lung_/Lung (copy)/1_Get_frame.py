from posixpath import split
import sys
import os
import glob
import cv2


# LOADING DATA #################################################################################
def video2frame(video_path, frame_save_path, time_interval):

	os.makedirs(frame_save_path) 
	
	vidcap = cv2.VideoCapture(video_path)
	success, image = vidcap.read()
	count = 0
	while success:
		count += 1
		if count % time_interval == 0:
			cv2.imencode('.jpg', image)[1].tofile(frame_save_path + "/%d.jpg" % count)
		success, image = vidcap.read()

	print(count)
	if success == 0:
		return 0

for arg in sys.argv:
    print(arg)

# arg 0 Get_frame.py
# arg[1] /home/lys/Lung/input_video  (The path of input video)
# arg[1] /home/lys/Lung/Frame  (The path of generated images)

list_of_video_file = glob.glob(os.path.join(sys.argv[1],"*.avi*"))
print(os.path)
print(list_of_video_file)

output_path=sys.argv[2]

assert len(list_of_video_file) > 0, "No video files were found in the input folder provided!!"


output_path

for video_filename in list_of_video_file:

	filename = video_filename.split('/')
	path_frame = filename[-1].replace('.avi','')
	frame_save_path = os.path.join('/home/lys/Lung/Frame/', path_frame)   
	video2frame(video_filename, frame_save_path,1)

