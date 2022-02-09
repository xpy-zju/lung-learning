# 把所有的json文件中 『图片名，圆圈信息』存入一个txt文件中
from asyncore import write
from ensurepip import version
from fnmatch import translate
import json
import os


def get_filename(path= 'output_frame/001',filetype ='.json'):
    name =[]
    final_name = []
    for root,dirs,files in os.walk(path):
        for i in files:
            if filetype in i:
                name.append(i.replace(filetype,''))#生成不带‘.json’后缀的文件名组成的列表
    final_name = [item + filetype for item in name]#生成‘.json’后缀的文件名组成的列表
    return final_name      #输出由有‘.json'后缀的文件名组成的列表
# print(get_filename(path, filetype))


# 读取列表所示的json文件，读出其中的 jpg名，圆圈label信息，并保存成txt文件
# def save_txt(final_name, json_path = 'output_frame/001/'):
#     for file in final_name:
#         file_path = json_path + file
#         with open(file_path) as f:
#             k = json.load(f)
#             k1 = []
#             k1.append(k['imagePath'])
#             for x in k['shapes']:
#                 # k1.append('point')
#                 for y in x['points']:
#                     new_list = [w/480 for w in y]  
#                     k1.append((str(new_list).replace("[",'')).replace(']',''))
#         print(k1)
#         with open('test.txt','a+') as t:
#             t.write((str(k1).replace("[",'')).replace(']','').replace("'",''))
#             t.write('\n')

# 读取列表所示的json文件，读出其中的 jpg名，以及最终所需标签
def save_label(final_name, json_path = 'output_frame/001/'):
    with open('label.csv','a+') as t:
        t.write('path,whole_num,x1,y1,r1,x2,y2,r2,x3,y3,r3')
        t.write('\n')
    for file in final_name:
        file_path = json_path + file
        with open(file_path) as f:
            k = json.load(f)
            k1 = []
            k1.append(json_path+k['imagePath'])
            if k['shapes'][0]["shape_type"] == "circle":
                k1.append(len(k['shapes'])/3)
            else:
                k1.append(0)
            for x in k['shapes']:
                if x["shape_type"] == "circle":
                    k1.append([w/480 for w in x['points'][0]])
                    # print('x0',x['points'][0])
                    # print('x1',x['points'][1])
                    k1.append((((x['points'][0][0]-x['points'][1][0])**2 + (x['points'][0][1]-x['points'][1][1])**2)**(0.5))/480)
            k1 = k1 + [0.0]*int((12-1.5*len(k1)))
        with open('label.csv','a+') as t:
            t.write((str(k1).replace("[",'')).replace(']','').replace("'",''))
            t.write('\n')
save_label(get_filename())