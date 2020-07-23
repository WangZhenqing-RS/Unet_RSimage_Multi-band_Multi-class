import numpy as np 
import os
import random
import gdal
import cv2

#  获取颜色字典
#  labelFolder 标签文件夹,之所以遍历文件夹是因为一张标签可能不包含所有类别颜色
#  classNum 类别总数(含背景)
def color_dict(labelFolder, classNum):
    colorDict = []
    #  获取文件夹内的文件名
    ImageNameList = os.listdir(labelFolder)
    for i in range(len(ImageNameList)):
        ImagePath = labelFolder + "/" + ImageNameList[i]
        img = cv2.imread(ImagePath).astype(np.uint32)
        #  如果是灰度，转成RGB
        if(len(img.shape) == 2):
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB).astype(np.uint32)
        #  为了提取唯一值，将RGB转成一个数
        img_new = img[:,:,0] * 1000000 + img[:,:,1] * 1000 + img[:,:,2]
        unique = np.unique(img_new)
        #  将第i个像素矩阵的唯一值添加到colorDict中
        for j in range(unique.shape[0]):
            colorDict.append(unique[j])
        #  对目前i个像素矩阵里的唯一值再取唯一值
        colorDict = sorted(set(colorDict))
        #  若唯一值数目等于总类数(包括背景)ClassNum，停止遍历剩余的图像
        if(len(colorDict) == classNum):
            break
    #  存储颜色的RGB字典，用于预测时的渲染结果
    colorDict_RGB = []
    for k in range(len(colorDict)):
        #  对没有达到九位数字的结果进行左边补零(eg:5,201,111->005,201,111)
        color = str(colorDict[k]).rjust(9, '0')
        #  前3位R,中3位G,后3位B
        color_RGB = [int(color[0 : 3]), int(color[3 : 6]), int(color[6 : 9])]
        colorDict_RGB.append(color_RGB)
    #  转为numpy格式
    colorDict_RGB = np.array(colorDict_RGB)
    #  存储颜色的GRAY字典，用于预处理时的onehot编码
    colorDict_GRAY = colorDict_RGB.reshape((colorDict_RGB.shape[0], 1 ,colorDict_RGB.shape[1])).astype(np.uint8)
    colorDict_GRAY = cv2.cvtColor(colorDict_GRAY, cv2.COLOR_BGR2GRAY)
    return colorDict_RGB, colorDict_GRAY

#  读取图像像素矩阵
#  fileName 图像文件名
def readTif(fileName):
    dataset = gdal.Open(fileName)
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    GdalImg_data = dataset.ReadAsArray(0, 0, width, height)
    return GdalImg_data

#  数据预处理：图像归一化+标签onehot编码
#  img 图像数据
#  label 标签数据
#  classNum 类别总数(含背景)
#  colorDict_GRAY 颜色字典
def dataPreprocess(img, label, classNum, colorDict_GRAY):
    #  归一化
    img = img / 255.0
    for i in range(colorDict_GRAY.shape[0]):
        label[label == colorDict_GRAY[i][0]] = i
    #  将数据厚度扩展到classNum层
    new_label = np.zeros(label.shape + (classNum,))
    #  将平面的label的每类，都单独变成一层
    for i in range(classNum):
        new_label[label == i,i] = 1                                          
    label = new_label
    return (img, label)

#  训练数据生成器
#  batch_size 批大小
#  train_image_path 训练图像路径
#  train_label_path 训练标签路径
#  classNum 类别总数(含背景)
#  colorDict_GRAY 颜色字典
#  resize_shape resize大小
def trainGenerator(batch_size, train_image_path, train_label_path, classNum, colorDict_GRAY, resize_shape = None):
    imageList = os.listdir(train_image_path)
    labelList = os.listdir(train_label_path)
    img = readTif(train_image_path + "\\" + imageList[0])
    #  GDAL读数据是(BandNum,Width,Height)要转换为->(Width,Height,BandNum)
    img = img.swapaxes(1, 0)
    img = img.swapaxes(1, 2)
    #  无限生成数据
    while(True):
        img_generator = np.zeros((batch_size, img.shape[0], img.shape[1], img.shape[2]), np.uint8)
        label_generator = np.zeros((batch_size, img.shape[0], img.shape[1]), np.uint8)
        if(resize_shape != None):
            img_generator = np.zeros((batch_size, resize_shape[0], resize_shape[1], resize_shape[2]), np.uint8)
            label_generator = np.zeros((batch_size, resize_shape[0], resize_shape[1]), np.uint8)
        #  随机生成一个batch的起点
        rand = random.randint(0, len(imageList) - batch_size)
        for j in range(batch_size):
            img = readTif(train_image_path + "\\" + imageList[rand + j])
            img = img.swapaxes(1, 0)
            img = img.swapaxes(1, 2)
            #  改变图像尺寸至特定尺寸(
            #  因为resize用的不多，我就用了OpenCV实现的，这个不支持多波段，需要的话可以用np进行resize
            if(resize_shape != None):
                img = cv2.resize(img, (resize_shape[0], resize_shape[1]))
            
            img_generator[j] = img
            
            label = readTif(train_label_path + "\\" + labelList[rand + j]).astype(np.uint8)
            #  若为彩色，转为灰度
            if(len(label.shape) == 3):
                label = label.swapaxes(1, 0)
                label = label.swapaxes(1, 2)
                label = cv2.cvtColor(label, cv2.COLOR_RGB2GRAY)
            if(resize_shape != None):
                label = cv2.resize(label, (resize_shape[0], resize_shape[1]))
            label_generator[j] = label
        img_generator, label_generator = dataPreprocess(img_generator, label_generator, classNum, colorDict_GRAY)
        yield (img_generator,label_generator)

#  测试数据生成器
#  test_iamge_path 测试数据路径
#  resize_shape resize大小
def testGenerator(test_iamge_path, resize_shape = None):
    imageList = os.listdir(test_iamge_path)
    for i in range(len(imageList)):
        img = readTif(test_iamge_path + "\\" + imageList[i])
        img = img.swapaxes(1, 0)
        img = img.swapaxes(1, 2)
        #  归一化
        img = img / 255.0
        if(resize_shape != None):
            #  改变图像尺寸至特定尺寸
            img = cv2.resize(img, (resize_shape[0], resize_shape[1]))
        #  将测试图片扩展一个维度,与训练时的输入[batch_size,img.shape]保持一致
        img = np.reshape(img, (1, ) + img.shape)
        yield img

#  保存结果
#  test_iamge_path 测试数据图像路径
#  test_predict_path 测试数据图像预测结果路径
#  model_predict 模型的预测结果
#  color_dict 颜色词典
def saveResult(test_image_path, test_predict_path, model_predict, color_dict, output_size):
    imageList = os.listdir(test_image_path)
    for i, img in enumerate(model_predict):
        channel_max = np.argmax(img, axis = -1)
        img_out = np.uint8(color_dict[channel_max.astype(np.uint8)])
        #  修改差值方式为最邻近差值
        img_out = cv2.resize(img_out, (output_size[0], output_size[1]), interpolation = cv2.INTER_NEAREST)
        #  保存为无损压缩png
        cv2.imwrite(test_predict_path + "\\" + imageList[i][:-4] + ".png", img_out)