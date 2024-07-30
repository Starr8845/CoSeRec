# 看一下原始sasrec、sasrec+cl、sasrec+几种不同的mask之后，高中低频物品表征分别有什么特点和变化
# 要统计：表征的秩的变化、表征的entropy、表征的norm、各个group里面表征的相似度
from models import SASRecModel
import argparse
import torch
import os
from utils import get_user_seqs, process_item_frequency
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt



parser = argparse.ArgumentParser()
#system args
parser.add_argument('--data_dir', default='./data/', type=str)
parser.add_argument('--output_dir', default='/home/zzx/seqRec/CLTrys/CoSeRec_Augment/CoSeRec/output/', type=str)
parser.add_argument('--data_name', default='Sports_and_Outdoors', type=str)
parser.add_argument('--do_eval', action='store_true')
parser.add_argument('--model_idx', default="temptempTry", type=str, help="model idenfier 10, 20, 30...")
parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")

#data augmentation args
parser.add_argument('--noise_ratio', default=0.0, type=float, \
                    help="percentage of negative interactions in a sequence - robustness analysis")
parser.add_argument('--training_data_ratio', default=1.0, type=float, \
                    help="percentage of training samples used for training - robustness analysis")
parser.add_argument('--augment_threshold', default=4, type=int, \
                    help="control augmentations on short and long sequences.\
                    default:-1, means all augmentations types are allowed for all sequences.\
                    For sequence length < augment_threshold: Insert, and Substitute methods are allowed \
                    For sequence length > augment_threshold: Crop, Reorder, Substitute, and Mask \
                    are allowed.")
parser.add_argument('--similarity_model_name', default='ItemCF_IUF', type=str, \
                    help="Method to generate item similarity score. choices: \
                    Random, ItemCF, ItemCF_IUF(Inverse user frequency), Item2Vec, LightGCN")
parser.add_argument("--augmentation_warm_up_epoches", type=float, default=160, \
                    help="number of epochs to switch from \
                    memory-based similarity model to \
                    hybrid similarity model.")
parser.add_argument('--base_augment_type', default='random', type=str, \
                    help="default data augmentation types. Chosen from: \
                    mask, crop, reorder, substitute, insert, random, \
                    combinatorial_enumerate (for multi-view).")
parser.add_argument('--augment_type_for_short', default='SIM', type=str, \
                    help="data augmentation types for short sequences. Chosen from: \
                    SI, SIM, SIR, SIC, SIMR, SIMC, SIRC, SIMRC.")
parser.add_argument("--tao", type=float, default=0.2, help="crop ratio for crop operator")
parser.add_argument("--gamma", type=float, default=0.7, help="mask ratio for mask operator")
parser.add_argument("--beta", type=float, default=0.2, help="reorder ratio for reorder operator") 
parser.add_argument("--substitute_rate", type=float, default=0.1, \
                    help="substitute ratio for substitute operator")
parser.add_argument("--insert_rate", type=float, default=0.4, \
                    help="insert ratio for insert operator")
parser.add_argument("--max_insert_num_per_pos", type=int, default=1, \
                    help="maximum insert items per position for insert operator - not studied")

## contrastive learning task args
parser.add_argument('--temperature', default= 1.0, type=float,
                    help='softmax temperature (default:  1.0) - not studied.')
parser.add_argument('--n_views', default=2, type=int, metavar='N',
                    help='Number of augmented data for each sequence - not studied.')

# model args
parser.add_argument("--model_name", default='CoSeRec', type=str)
parser.add_argument("--hidden_size", type=int, default=64, help="hidden size of transformer model")
parser.add_argument("--num_hidden_layers", type=int, default=2, help="number of layers")
parser.add_argument('--num_attention_heads', default=2, type=int)
parser.add_argument('--hidden_act', default="gelu", type=str) # gelu relu
parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.5, help="attention dropout p")
parser.add_argument("--hidden_dropout_prob", type=float, default=0.5, help="hidden dropout p")
parser.add_argument("--initializer_range", type=float, default=0.02)
parser.add_argument('--max_seq_length', default=50, type=int)

# train args
parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
parser.add_argument("--batch_size", type=int, default=256, help="number of batch_size")
parser.add_argument("--epochs", type=int, default=300, help="number of epochs")
parser.add_argument("--no_cuda", action="store_true")
parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
parser.add_argument("--seed", default=1, type=int)
parser.add_argument("--cf_weight", type=float, default=0.1, \
                    help="weight of contrastive learning task")
parser.add_argument("--rec_weight", type=float, default=1.0, \
                    help="weight of contrastive learning task")

#learning related
parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay of adam")
parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")

parser.add_argument("--cl", action="store_true")
parser.add_argument("--multi_neg", action="store_true")
parser.add_argument("--use_freq", action="store_true")

parser.add_argument('--mask_strategy', default='random', type=str, \
                    help="random, mask_high, mask_mid_low")


args = parser.parse_args()
    
args.data_file = args.data_dir + args.data_name + '.txt'
user_seq, max_item, valid_rating_matrix, test_rating_matrix = \
        get_user_seqs(args.data_file)
args.item_size = max_item + 2
args_str = f'{args.model_name}-{args.data_name}-{args.model_idx}'

checkpoint = args_str + '.pt'
args.checkpoint_path = os.path.join(args.output_dir, checkpoint)
item_frequency, (high_freq, mid_freq, low_freq), item_freq_class = process_item_frequency(args.data_file) 
args.item_freq_class = item_freq_class
args.cuda_condition = torch.cuda.is_available() and not args.no_cuda
model = SASRecModel(args=args)
model.load_state_dict(torch.load(args.checkpoint_path))


# 下面开始分析模型
# 拿出来item_embedding
high_item_emb = model.get_item_embeddings(torch.tensor(high_freq)).detach()
mid_item_emb = model.get_item_embeddings(torch.tensor(mid_freq)).detach()
low_item_emb = model.get_item_embeddings(torch.tensor(low_freq)).detach()


# def cal_avg_norm(emb):
#     norm = torch.norm(emb, dim=1)
#     return torch.mean(norm)

# high_norm = cal_avg_norm(high_item_emb)
# mid_norm = cal_avg_norm(mid_item_emb)
# low_norm = cal_avg_norm(low_item_emb)
# print(high_norm)
# print(mid_norm)
# print(low_norm)

# def cal_avg_sim(emb):
#     item_emb_normalized = F.normalize(emb, p=2, dim=1)
#     similarity_matrix = torch.matmul(item_emb_normalized, item_emb_normalized.T)
#     # 排除对角线上的余弦相似度（每个向量与自己的相似度）
#     batch_size = high_item_emb.size(0)
#     eye = torch.eye(batch_size, device=high_item_emb.device)
#     similarity_matrix = similarity_matrix * (1 - eye)
#     average_cosine_similarity = similarity_matrix.sum() / (batch_size * (batch_size - 1))
#     return average_cosine_similarity
# high_sim = cal_avg_sim(high_item_emb)
# mid_sim = cal_avg_sim(mid_item_emb)
# low_sim = cal_avg_sim(low_item_emb)
# print(high_sim)
# print(mid_sim)
# print(low_sim)

def svd_values(repres):
    def svd(repres):
        c = np.linalg.svd(repres)
        return c[1]
    svds_cl_all = svd(repres)
    svds_cl_all_normalized = svds_cl_all/np.max(svds_cl_all)
    return svds_cl_all_normalized
high_svd = svd_values(high_item_emb)
mid_svd = svd_values(mid_item_emb)
low_svd = svd_values(low_item_emb)

plt.plot(high_svd,label='High SVD')
plt.plot(mid_svd,label='Mid SVD')
plt.plot(low_svd,label='Low SVD')
# 添加图例
plt.legend()

plt.savefig("/home/zzx/seqRec/CLTrys/CoSeRec_Augment/CoSeRec/img/"+args.model_idx+"svd.jpg")






# single_1024_2
# single_cl_1024
# single_cl_mask_mid_low_1024
# single_cl_mask_1024
# 

