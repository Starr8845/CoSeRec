from tensorboard.backend.event_processing import event_accumulator
from torch.utils.tensorboard import SummaryWriter  

import os
input_path_folder = '/home/zzx/seqRec/CLTrys/CoSeRec/logs/Sports_and_Outdoors/CoSeRec_single_single_cl_mask_mid_low_1024'
output_path = '/home/zzx/seqRec/CLTrys/CoSeRec/logs/Sports_and_Outdoors/CoSeRec_single_single_cl_mask_mid_low_all_1024'  # 输出只需要指定文件夹即可
 
writer = SummaryWriter(output_path)  # 创建一个SummaryWriter对象
input_path = input_path_folder
# 读取需要修改的event文件
ea = event_accumulator.EventAccumulator(input_path)
ea.Reload()
tags = ea.scalars.Keys()  # 获取所有scalar中的keys
print(tags)
# 写入新的文件

for tag in tags:
    scalar_list = ea.scalars.Items(tag)
    print(tag, scalar_list)
    
    flag=False
    for scalar in scalar_list:
        if scalar.step==0 and flag==True:
            break
        if scalar.step==0 and flag==False:
            flag=True
        print(scalar.step)
        writer.add_scalar(tag, scalar.value, scalar.step, scalar.wall_time)  # 添加修改后的值到新的event文件中

writer.close()  # 关闭SummaryWriter对象


