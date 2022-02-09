# 肺部支气管图像识别V1.0
> 期望实现目标:训练一个目标检测网络, 输入支气管的图像,输出图像中分岔支气管的位置及大小信息,为后续的学习导航等实际应用场景做准备

## 项目环境及框架
-  ubuntu20.04,pytorch-GPU, CUDA 11.3
-  医用支气管镜从肺部硅胶模型获取训练数据
-  labelme打标签,标签形式暂定[path,whole_num,x1,y1,r1,x2,y2,r2,x3,y3,r3]
-  resnet18欲训练模型 + 修改最后一层全连接层, 冻结卷积层参数训练
-  对数据集7/2/1分为,train/val/test

## 文件解释
> - input_video存放视频
> - output_frame/001视频生成的图片及标签json文件
> - output_frame/data_dataset_voc，json文件转voc格式，主要是一些合成的png和xml文件，暂时用不到
> - output_frame/labelme2voc.py作用见上一条
> - output_frame/labels.txt，labelme2voc.py所需要的标签类别文件

> - json2dataset.py 将json标签按需求批量生成csv格式
> - dataloader.py 读取csv并划分list，打包dataset类，准备送入dataloader
> - resnet.py 训练函数，仍需改进

### 操作流程及完成情况

> 1. 视频文件并逐帧取出图片 Get_frame.py
> 2. labelme打标签,生成json文件 
> 3. json文件批量转csv文件
> 4. 训练网络并保存网络，还没做loss可视化之类的
> 5. 在test上测试，由于暂时无法检验accuracy，因此先采用保存网络，输入测试集并可视化的方法

![avatar](/1.jpg)

