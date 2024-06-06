from collections import defaultdict


with open("/home/zzx/seqRec/CLTrys/CoSeRec/data/ml-1m_pre.txt", "r") as f:
    lines = f.readlines()

data = defaultdict(list)

for line in lines:
    line = line.strip().split(' ')
    data[line[0]].append(line[1])
# print(data)


with open("/home/zzx/seqRec/CLTrys/CoSeRec/data/ml-1m.txt", "w") as f:
    for num, seq in data.items():
        f.write(num+' '+' '.join(seq)+'\n')