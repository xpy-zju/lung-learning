from posixpath import split
import sys
import os
import glob
import cv2
import os
import random
import string

# LOADING DATA #################################################################################
def save_image(image, frame_save_path, count, mode):
	if mode == 'train':
		salt = ''.join(random.sample(string.ascii_letters + string.digits, 8))  # 随机输出8位由英文字符和数字组成的字符串
		cv2.imencode('.jpg', image)[1].tofile(frame_save_path + "/%s.jpg" % salt)
	
	if mode == 'test':
		if count < 10:
			cv2.imencode('.jpg', image)[1].tofile(frame_save_path +"/000"+"%d.jpg" % count)
		if count >= 10 and count < 100:
			cv2.imencode('.jpg', image)[1].tofile(frame_save_path +"/00"+"%d.jpg" % count)
		if count >= 100 and count < 1000:
			cv2.imencode('.jpg', image)[1].tofile(frame_save_path +"/0"+"%d.jpg" % count)
		if count >= 1000:
			cv2.imencode('.jpg', image)[1].tofile(frame_save_path +"/%d.jpg" % count)
	return				

def video2frame(video_path, frame_save_path, time_interval, len=10000, mode='train'):  # len: test dataset lenth

	os.makedirs(frame_save_path) 
	
	vidcap = cv2.VideoCapture(video_path)
	success, image = vidcap.read()
	count = 0
	while success:		
		count += 1
		if count == len and mode == 'test':    # for test dataset
			break
		if count % time_interval == 0:
			save_image(image, frame_save_path, count, mode)
		success, image = vidcap.read()

	print(count)
	if success == 0:
		return 0





# for arg in sys.argv:
#     print(arg)
# arg 0 Get_frame.py
# arg[1] /home/lys/Lung/input_video  (The path of input video)
# arg[2] /home/lys/Lung/output_Frame  (The path of generated images)
# arg[3] test or train

list_of_video_file = glob.glob(os.path.join(sys.argv[1],"*.avi*"))
output_path=sys.argv[2]
Mode=sys.argv[3]
Length=0

print('1 Local path ',os.path)
print('2 Input path ',list_of_video_file)
print('3 Generate Mode', Mode)


assert len(list_of_video_file) > 0, "No video files were found in the input folder provided!!"
for video_filename in list_of_video_file:

	filename = video_filename.split('/')
	path_frame = filename[-1].replace('.avi','')
	
	if Mode == 'train':
		frame_save_path = os.path.join('/home/lys/Lung_my/output_frame/', path_frame)   
	else:
		frame_save_path = '/home/lys/Lung_my/output_frame/test'
		Length = 200
	video2frame(video_filename, frame_save_path,1, Length, Mode)

