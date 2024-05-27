# 读一下数据集， 分析一下item的稀疏度
import numpy as np
from collections import defaultdict

dataset_name = "Sports_and_Outdoors"


with open(f"/home/zzx/seqRec/CLTrys/CoSeRec/data/{dataset_name}.txt","r") as f:
    raw_data = f.readlines()


Seqs = [each.strip().split(' ')[1:] for each in raw_data]

print("序列总个数",len(Seqs))

lens = [len(each) for each in Seqs]

print("序列平均长度", np.mean(lens))

print("序列最大长度", np.max(lens), "; 序列最小长度", np.min(lens))

# TODO：这里要画一个序列长度的分布图

# 下面要统计一下item 出现次数的分布

item_frequency = defaultdict(int)

for seq in Seqs:
    for each_item in seq:
        item_frequency[each_item] +=1

print(item_frequency)



