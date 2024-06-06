import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

def build_sim_graph(cf_graph, k, user_num, item_num):
    import dgl
    
    similarity = cosine_similarity(cf_graph.transpose())
    # filter topk connections
    sim_items_slices = []
    sim_weights_slices = []
    i = 0
    print(similarity.shape) # [3417, 3417]
    while i < similarity.shape[0]:     
        end = min(similarity.shape[0], i+256)
        print(i, end)
        sim = similarity[i:end, :] # 改一下
        sim_items = np.argpartition(sim, -(k+1), axis=1)[:, -(k+1):]
        sim_weights = np.take_along_axis(sim, sim_items, axis=1)
        sim_items_slices.append(sim_items)
        sim_weights_slices.append(sim_weights)
        i = i + 256

    sim_items = np.concatenate(sim_items_slices, axis=0)
    sim_weights = np.concatenate(sim_weights_slices, axis=0)
    row = []
    col = []
    for i in range(len(sim_items)):
        row.extend([i]*len(sim_items[i]))
        col.extend(sim_items[i])
    # values = sim_weights / sim_weights.sum(axis=1, keepdims=True) # 这一行先暂时注释掉
    row = np.array(row)
    col = np.array(col)
    
    values = sim_weights
    
    values = np.nan_to_num(values).flatten()
    # 把权重为0的边删掉
    row = row[values>0]
    col = col[values>0]
    values = values[values>0]
    adj_mat = csr_matrix((values, (row, col)), shape=(
        item_num + 1, item_num + 1))
    g = dgl.from_scipy(adj_mat, 'w')
    g.edata['w'] = g.edata['w'].float()
    return g

def generate_rating_matrix_valid(user_seq, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in user_seq.items():
        for item in item_list:
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items), dtype=np.float32)

    return rating_matrix

# 数据读进来
import numpy as np
from collections import defaultdict

dataset_name = "ml-1m"


with open(f"/home/zzx/seqRec/CLTrys/CoSeRec/data/{dataset_name}.txt","r") as f:
    raw_data = f.readlines()

user_seq = {}
for each in raw_data:
    each = [int(temp) for temp in each.strip().split(' ')]
    user_seq[each[0]] = each[1:]


item_frequency = defaultdict(int)

for seq in user_seq.values():
    for each_item in seq:
        item_frequency[each_item] +=1


rating_matrix = generate_rating_matrix_valid(user_seq, len(user_seq)+1, len(item_frequency)+1)
# 这里rating matrix有问题，需要仔细看一下
# print(rating_matrix)
print(type(rating_matrix))

print(rating_matrix.count_nonzero())
print(rating_matrix.shape)

g = build_sim_graph(rating_matrix, 20, len(user_seq), len(item_frequency))


import torch 
print(g)
# 算一下边的权重   cosine similarity
print(g.edata)
print(torch.mean(g.edata['w']))
temp_weights = g.edata['w']
print(torch.mean(temp_weights[temp_weights<1.]))
# 


# 下面要看一下 low frequency 的 item的边的分布

item_fre_threshhold = np.average(list(item_frequency.values()))
print(item_fre_threshhold)
tail_nodes = np.array([node for node, freq in item_frequency.items() if freq<item_fre_threshhold])

print("长尾节点平均度, ", torch.mean(g.in_degrees(tail_nodes)+g.out_degrees(tail_nodes)))

print(g.in_degrees([i for i in range(1, len(item_frequency)+1)]))

# 计算一下全局节点的degress
print()
