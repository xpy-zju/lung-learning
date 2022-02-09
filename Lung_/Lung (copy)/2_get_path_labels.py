import os
import numpy as np
import pickle

root = '/home/lys/Lung/'                                    #Work dir
video_dir = os.path.join(root, 'input_video')               #Video
img_dir = os.path.join(root, 'Frame')                      #Images
phase_dir = os.path.join(root, 'Description')               #Labels

print(video_dir)
print(img_dir)
print(phase_dir)


# Frames目录下的文件（001/002...）==============cha2021==================
def get_dirs(source_dir, gen_dir):                             # 当前目录是文件夹 /Frames/001
    file_paths = []
    file_names = []
    for lists in os.listdir(source_dir):    
        lists = lists.replace('.avi', '')          
        path = os.path.join(gen_dir, lists)   

        if os.path.isdir(path):
            file_paths.append(path)
            file_names.append(os.path.basename(path))           # os.path.basename(path) -> 001
    file_names.sort(key=lambda x: int(x))
    file_paths.sort(key=lambda x: int(os.path.basename(x)))
    return file_names, file_paths

# Example
#   a, b = get_dirs(video_dir, img_dir)
#   print(a)               ['001']              
#   print(b)               ['/home/lys/Lung/Frame/001']



############################### From here go on ###################################
def get_files(root_dir):                           # 当前目录不是文件夹 /Procedural_description/001.txt
    file_paths = []
    file_names = []
    for lists in os.listdir(root_dir):
        path = os.path.join(root_dir, lists)
        if not os.path.isdir(path):
            file_paths.append(path)
            file_names.append(os.path.basename(path))
    file_names.sort()
    file_paths.sort()
    return file_names, file_paths


img_dir_names, img_dir_paths = get_dirs(img_dir)
phase_file_names, phase_file_paths = get_files(phase_dir)

# Phase catalog ========================
# Transfer Left to Right :  L2R
# Transfer Right to Left :  R2L

phase_dict = {}
phase_dict_key = ['Idle', 'Block 1 L2R', 'Block 2 L2R', 'Block 3 L2R', 'Block 4 L2R', 'Block 5 L2R', 'Block 6 L2R',
                          'Block 1 R2L', 'Block 2 R2L', 'Block 3 R2L', 'Block 4 R2L', 'Block 5 R2L', 'Block 6 R2L']
for i in range(len(phase_dict_key)):
    phase_dict[phase_dict_key[i]] = i
print(phase_dict)


# 读取并构建所有视频的信息 ==================
all_info_all = []

for j in range(len(phase_file_names)):
    #采样速率
    downsample_rate = 1

    phase_file = open(phase_file_paths[j])   #../Procedural_description/第j个

    #video_num_file = int(os.path.splitext(os.path.basename(phase_file_paths[j]))[0][5:7])
    #video_num_dir = int(os.path.basename(img_dir_paths[j]))
    #print("video_num_file:", video_num_file, "video_num_dir:", video_num_dir, "rate:", downsample_rate)
    info_all = []
    first_line = True
    for phase_line in phase_file:
        phase_split = phase_line.split('\t')      # 空格分隔
        if first_line:
            first_line = False
            continue
        if int(phase_split[0]) % downsample_rate == 0:
            info_each = []                              # info_each = [img_path, annotation]
            img_file_each_path = os.path.join(img_dir_paths[j], 'frame' + str(int(phase_split[0])+1) + '.jpg')  #video
            info_each.append(img_file_each_path)
            info_each.append(phase_dict[phase_split[2]])       # 取phase
            info_all.append(info_each)

    all_info_all.append(info_all)

# video_step
with open('./cha2021_2.pkl', 'wb') as f:
    pickle.dump(all_info_all, f)
with open('./cha2021_2.pkl', 'rb') as f:
    all_info_149 = pickle.load(f)

# 分成 train + val + test ==================
train_file_paths = []
val_file_paths = []
train_labels = []
val_labels = []

train_num_each = []    # 每个train video的帧数， len(all_info_149[i]) -> sum
val_num_each = []

num_train = 60      # 训练集 数目
num_val = 30       # 测试集 数目

stat = np.zeros(13).astype(int)    # 不同动作计数
#split 1 
#for i in range(num_train):
#split 2
for i in range(30,90):
    train_num_each.append(len(all_info_149[i]))          # 训练集的video数目  all_in_all
    for j in range(len(all_info_149[i])):                # 第i个video的参数   info_all [path, annotation] * num_zhen
        train_file_paths.append(all_info_149[i][j][0])   # path
        train_labels.append(all_info_149[i][j][1:])      # 去掉列表中的第一个元素 的意思 -- annotation
        stat[all_info_149[i][j][1]] += 1
print("-------------------Train stage-------------------")
print("Train_file_paths: " + str(len(train_file_paths)))
print("Train_labels: " + str(len(train_labels)))
print("Status:")
print(stat)

for i in range(0,30):
    val_num_each.append(len(all_info_149[i]))
    for j in range(len(all_info_149[i])):
        val_file_paths.append(all_info_149[i][j][0])
        val_labels.append(all_info_149[i][j][1:])
print("-------------------Val stage-------------------")
print("Val_file_paths: " + str(len(val_file_paths)))
print("Val_labels: " + str(len(val_labels)))


# loading data ==================

loading_data = []

loading_data.append(train_file_paths)   # 0
loading_data.append(train_labels)       # 1
loading_data.append(train_num_each)     # 2

loading_data.append(val_file_paths)    # 3
loading_data.append(val_labels)        # 4
loading_data.append(val_num_each)      # 5

#Video
with open('Res18_split2.pkl', 'wb') as f:
    pickle.dump(loading_data, f)


print('Done')
print()
