import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from seg_unet import unet
#from Model.seg_hrnet import seg_hrnet
from dataProcess import trainGenerator, color_dict
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import datetime
import xlwt
import os


'''
数据集相关参数
'''
#  训练数据图像路径
train_image_path = "Data\\train\\image"
#  训练数据标签路径
train_label_path = "Data\\train\\label"
#  验证数据图像路径
validation_image_path = "Data\\validation\\image"
#  验证数据标签路径
validation_label_path = "Data\\validation\\label"

'''
模型相关参数
'''
#  批大小
batch_size = 2
#  类的数目(包括背景)
classNum = 2
#  模型输入图像大小
input_size = (512, 512, 3)
#  训练模型的迭代总轮数
epochs = 50
#  初始学习率
learning_rate = 1e-4
#  预训练模型地址
premodel_path = None
#  训练模型保存地址
model_path = "Model\\unet_model.hdf5"

#  训练数据数目
train_num = len(os.listdir(train_image_path))
#  验证数据数目
validation_num = len(os.listdir(validation_image_path))
#  训练集每个epoch有多少个batch_size
steps_per_epoch = train_num / batch_size
#  验证集每个epoch有多少个batch_size
validation_steps = validation_num / batch_size
#  标签的颜色字典,用于onehot编码
colorDict_RGB, colorDict_GRAY = color_dict(train_label_path, classNum)


#  得到一个生成器，以batch_size的速率生成训练数据
train_Generator = trainGenerator(batch_size,
                                 train_image_path, 
                                 train_label_path,
                                 classNum ,
                                 colorDict_GRAY,
                                 input_size)

#  得到一个生成器，以batch_size的速率生成验证数据
validation_data = trainGenerator(batch_size,
                                 validation_image_path,
                                 validation_label_path,
                                 classNum,
                                 colorDict_GRAY,
                                 input_size)
#  定义模型
model = unet(pretrained_weights = premodel_path, 
             input_size = input_size, 
             classNum = classNum, 
             learning_rate = learning_rate)
#model = seg_hrnet(pretrained_weights = premodel_path, 
#                  input_size = input_size, 
#                  classNum = classNum, 
#                  learning_rate = learning_rate)
#  打印模型结构
model.summary()
#  回调函数
#  val_loss连续10轮没有下降则停止训练
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 10)
#  当3个epoch过去而val_loss不下降时，学习率减半
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = 3, verbose = 1)
model_checkpoint = ModelCheckpoint(model_path,
                                   monitor = 'loss',
                                   verbose = 1,# 日志显示模式:0->安静模式,1->进度条,2->每轮一行
                                   save_best_only = True)

#  获取当前时间
start_time = datetime.datetime.now()

#  模型训练
history = model.fit_generator(train_Generator,
                    steps_per_epoch = steps_per_epoch,
                    epochs = epochs,
                    callbacks = [early_stopping, model_checkpoint, model_checkpoint],
                    validation_data = validation_data,
                    validation_steps = validation_steps)

#  训练总时间
end_time = datetime.datetime.now()
log_time = "训练总时间: " + str((end_time - start_time).seconds / 60) + "m"
time = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d-%H%M%S')
print(log_time)
with open('TrainTime_%s.txt'%time,'w') as f:
    f.write(log_time)
    
#  保存并绘制loss,acc
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
book = xlwt.Workbook(encoding='utf-8', style_compression=0)
sheet = book.add_sheet('test', cell_overwrite_ok=True)
for i in range(len(acc)):
    sheet.write(i, 0, acc[i])
    sheet.write(i, 1, val_acc[i])
    sheet.write(i, 2, loss[i])
    sheet.write(i, 3, val_loss[i])
book.save(r'AccAndLoss_%s.xls'%time)
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'r', label = 'Training acc')
plt.plot(epochs, val_acc, 'b', label = 'Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig("accuracy_%s.png"%time, dpi = 300)
plt.figure()
plt.plot(epochs, loss, 'r', label = 'Training loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig("loss_%s.png"%time, dpi = 300)
plt.show()
