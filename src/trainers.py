# -*- coding: utf-8 -*-

import numpy as np
from tqdm import tqdm
import random

import torch
import torch.nn as nn
from torch.optim import Adam

import dgl

from torch.utils.data import DataLoader, RandomSampler
from datasets import RecWithContrastiveLearningDataset
from modules import NCELoss, NTXent
from utils import recall_at_k, ndcg_k, get_metric, get_user_seqs, nCr

class Trainer:
    def __init__(self, model, train_dataloader,
                 eval_dataloader,
                 test_dataloader, 
                 args):

        self.args = args
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")

        self.model = model
        self.online_similarity_model = args.online_similarity_model

        self.total_augmentaion_pairs = nCr(self.args.n_views, 2)
        #projection head for contrastive learn task
        self.projection = nn.Sequential(nn.Linear(self.args.max_seq_length*self.args.hidden_size, \
                                        512, bias=False), nn.BatchNorm1d(512), nn.ReLU(inplace=True), 
                                        nn.Linear(512, self.args.hidden_size, bias=True))
        if self.cuda_condition:
            self.model.cuda()
            self.projection.cuda()
        # Setting the train and test data loader
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader

        # self.data_name = self.args.data_name
        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optim = Adam(self.model.parameters(), lr=self.args.lr, betas=betas, weight_decay=self.args.weight_decay)

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

        self.cf_criterion = NCELoss(self.args.temperature, self.device)
        # self.cf_criterion = NTXent()
        print("self.cf_criterion:", self.cf_criterion.__class__.__name__)
        
    def __refresh_training_dataset(self, item_embeddings):
        """
        use for updating item embedding
        """
        user_seq, _, _, _ = get_user_seqs(self.args.data_file)
        self.args.online_similarity_model.update_embedding_matrix(item_embeddings)
        # training data for node classification
        train_dataset = RecWithContrastiveLearningDataset(self.args, user_seq, 
                                        data_type='train', similarity_model_type='hybrid')
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=self.args.batch_size)
        return train_dataloader
        
    def train(self, epoch):
        # start to use online item similarity
        # zhangzexi: 暂时忽略 online item similarity这一个分支
        # if epoch > self.args.augmentation_warm_up_epoches:
        #     print("refresh dataset with updated item embedding")
        #     self.train_dataloader = self.__refresh_training_dataset(self.model.item_embeddings)
        self.iteration(epoch, self.train_dataloader)

    def valid(self, epoch, full_sort=False):
        return self.iteration(epoch, self.eval_dataloader, full_sort=full_sort, train=False)

    def test(self, epoch, full_sort=False):
        return self.iteration(epoch, self.test_dataloader, full_sort=full_sort, train=False)

    def iteration(self, epoch, dataloader, full_sort=False, train=True):
        raise NotImplementedError

    def get_sample_scores(self, epoch, pred_list):
        pred_list = (-pred_list).argsort().argsort()[:, 0]
        HIT_1, NDCG_1, MRR = get_metric(pred_list, 1)
        HIT_5, NDCG_5, MRR = get_metric(pred_list, 5)
        HIT_10, NDCG_10, MRR = get_metric(pred_list, 10)
        post_fix = {
            "Epoch": epoch,
            "HIT@1": '{:.4f}'.format(HIT_1), "NDCG@1": '{:.4f}'.format(NDCG_1),
            "HIT@5": '{:.4f}'.format(HIT_5), "NDCG@5": '{:.4f}'.format(NDCG_5),
            "HIT@10": '{:.4f}'.format(HIT_10), "NDCG@10": '{:.4f}'.format(NDCG_10),
            "MRR": '{:.4f}'.format(MRR),
        }
        print(post_fix)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')
        return [HIT_1, NDCG_1, HIT_5, NDCG_5, HIT_10, NDCG_10, MRR], str(post_fix)

    def get_full_sort_score(self, epoch, answers, pred_list, name=""):
        recall, ndcg = [], []
        for k in [5, 10, 15, 20]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        post_fix = {
            "Epoch": epoch,
            "name": name,
            "HIT@5": '{:.4f}'.format(recall[0]), "NDCG@5": '{:.4f}'.format(ndcg[0]),
            "HIT@10": '{:.4f}'.format(recall[1]), "NDCG@10": '{:.4f}'.format(ndcg[1]),
            "HIT@20": '{:.4f}'.format(recall[3]), "NDCG@20": '{:.4f}'.format(ndcg[3])
        }
        print(post_fix)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')
        return [recall[0], ndcg[0], recall[1], ndcg[1], recall[3], ndcg[3]], str(post_fix)

    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name))

    def cross_entropy(self, seq_out, pos_ids, neg_ids):
        # seq_out: [bs, 64]
        # pos_emb = self.model.item_embeddings(pos_ids)
        # neg_emb = self.model.item_embeddings(neg_ids)
        pos_emb = self.model.get_item_embeddings(pos_ids) # [bs, 1, 64]
        neg_emb = self.model.get_item_embeddings(neg_ids) # [bs, 1, 64]
        
        # [batch*seq_len hidden_size]
        pos = pos_emb.view(-1, pos_emb.size(2)) # [bs, 64]
        neg = neg_emb.view(-1, neg_emb.size(2)) # [bs, 64]
        
        pos_logits = torch.sum(pos * seq_out, -1) # [batch*seq_len]
        neg_logits = torch.sum(neg * seq_out, -1)
        logits = - torch.log(torch.sigmoid(pos_logits) + 1e-24) - torch.log(1 - torch.sigmoid(neg_logits) + 1e-24)
        loss = torch.mean(
            logits
        )
        return logits.detach(), loss

    def cross_entropy_2(self, seq_out, pos_ids, neg_ids):
        pos_emb = self.model.get_item_embeddings(pos_ids)
        neg_emb = self.model.get_item_embeddings(neg_ids)
        pos = pos_emb.view(-1, pos_emb.size(2)) #[256, 64]
        neg = neg_emb.view(-1, neg_emb.size(2)) #[256, 64]
        seq_emb = seq_out.view(-1, self.args.hidden_size) #[256, 64]
        return self.cf_criterion(seq_emb, pos)

    def predict_sample(self, seq_out, test_neg_sample):
        # [batch 100 hidden_size]
        # test_item_emb = self.model.item_embeddings(test_neg_sample)
        test_item_emb = self.model.get_item_embeddings(test_neg_sample)
        
        # [batch hidden_size]
        test_logits = torch.bmm(test_item_emb, seq_out.unsqueeze(-1)).squeeze(-1)  # [B 100]
        return test_logits

    def predict_full(self, seq_out):
        # [item_num hidden_size]
        # test_item_emb = self.model.item_embeddings.weight
        test_item_emb = self.model.get_item_embeddings()
        # [batch hidden_size ]
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred

class CoSeRecTrainer(Trainer):

    def __init__(self, model,
                 train_dataloader,
                 eval_dataloader,
                 test_dataloader, 
                 args,
                 writer):
        super(CoSeRecTrainer, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader, 
            args
        )
        self.writer = writer

    def _one_pair_contrastive_learning(self, inputs):
        '''
        contrastive learning given one pair sequences (batch)
        inputs: [batch1_augmented_data, batch2_augmentated_data]
        '''
        cl_batch = torch.cat(inputs, dim=0)
        cl_batch = cl_batch.to(self.device)
        cl_sequence_output = self.model.transformer_encoder(cl_batch)
        # cf_sequence_output = cf_sequence_output[:, -1, :]
        cl_sequence_flatten = cl_sequence_output.view(cl_batch.shape[0], -1)
        # cf_output = self.projection(cf_sequence_flatten)
        batch_size = cl_batch.shape[0]//2
        cl_output_slice = torch.split(cl_sequence_flatten, batch_size)
        cl_loss = self.cf_criterion(cl_output_slice[0], 
                                cl_output_slice[1])
        return cl_loss

    def item_info_NCE(self, src_ids, dst_ids):
        src_emb = self.model.item_embeddings(src_ids)
        dst_emb = self.model.item_embeddings(dst_ids)
        
        # 为了防止显存爆掉，batch_size手动设置一下
        batch_size = 256
        src_emb_list = torch.split(src_emb, batch_size)
        dst_emb_list = torch.split(dst_emb, batch_size)

        cl_loss = 0.0
        for i in range(len(src_emb_list)):
            cl_loss += self.cf_criterion(src_emb_list[i], 
                                dst_emb_list[i])
        return cl_loss/len(src_emb_list)

    def iteration(self, epoch, dataloader, full_sort=True, train=True):

        str_code = "train" if train else "test"

        # Setting the tqdm progress bar

        if train:
            self.model.train()
            rec_avg_loss = 0.0
            cl_individual_avg_losses = [0.0 for i in range(self.total_augmentaion_pairs)]
            cl_sum_avg_loss = 0.0
            joint_avg_loss = 0.0
            itemcl_sum_avg_loss = 0.0

            print(f"rec dataset length: {len(dataloader)}")
            rec_cf_data_iter = tqdm(enumerate(dataloader), total=len(dataloader))

            for i, (rec_batch, cl_batches) in rec_cf_data_iter:
            # for i, rec_batch in rec_cf_data_iter:
                '''
                rec_batch shape: key_name x batch_size x feature_dim
                cl_batches shape: 
                    list of n_views x batch_size x feature_dim tensors
                '''
                # 0. batch_data will be sent into the device(GPU or CPU)
                rec_batch = tuple(t.to(self.device) for t in rec_batch)
                _, input_ids, target_pos, target_neg = rec_batch

                # ---------- recommendation task ---------------#
                sequence_output_all_pos = self.model.transformer_encoder(input_ids, all_pos=True)
                sequence_output = sequence_output_all_pos[:, -1, :]
                if self.args.multi_neg:
                    rec_loss = self.cross_entropy_2(sequence_output, target_pos, target_neg)
                else:
                    rec_logits, rec_loss = self.cross_entropy(sequence_output, target_pos, target_neg)
                # zzx: sequence_output: [bs, 64]
                # target_pos, target_neg: [bs, 1]

                # 分开看一下 在不同类别的target item的样本上，分别的loss情况

                if i == 0:
                    pred_logits_list = rec_logits.cpu().data.numpy()
                    answer_list = target_pos.cpu().data.numpy()
                else:
                    pred_logits_list = np.append(pred_logits_list, rec_logits.cpu().data.numpy(), axis=0)
                    answer_list = np.append(answer_list, target_pos.cpu().data.numpy(), axis=0)

                
                joint_loss = 0

                # ---------- contrastive learning task -------------#
                cl_losses = []
                if self.args.cl:
                    for cl_batch in cl_batches:
                        cl_loss = self._one_pair_contrastive_learning(cl_batch)
                        cl_losses.append(cl_loss)
                
                # 在item表征 上加一个item 表征的约束 应用对比学习损失
                # 先把这个对比损失注释掉  没有效果
                # if self.args.item_graph is not None:
                #     # target_pos
                #     nonzero_indices = torch.nonzero(target_pos)
                #     item_ids = target_pos[nonzero_indices[:, 0], nonzero_indices[:, 1]]
                #     item_ids = item_ids.view(-1)

                #     num_neighbors = 2
                #     sampled_graph = dgl.sampling.sample_neighbors(self.args.item_graph, item_ids, num_neighbors, edge_dir="out")
                #     # 获取采样后的邻居
                #     sampled_edges = sampled_graph.edges()
                #     src, dst = sampled_edges
                #     # 形成一个infoNCE 损失
                #     loss_item_cl = self.item_info_NCE(src, dst)
                #     # 更好的方式是 形成一个待检索的dictionary
                #     # 
                #     joint_loss += 0.1*loss_item_cl
                #     itemcl_sum_avg_loss += loss_item_cl.item()

                joint_loss += self.args.rec_weight * rec_loss
                for cl_loss in cl_losses:
                    joint_loss += self.args.cf_weight * cl_loss
                self.optim.zero_grad()
                joint_loss.backward()
                self.optim.step()

                rec_avg_loss += rec_loss.item()

                for i, cl_loss in enumerate(cl_losses):
                    cl_individual_avg_losses[i] += cl_loss.item()
                    cl_sum_avg_loss += cl_loss.item()
                joint_avg_loss += joint_loss.item()


            post_fix = {
                "epoch": epoch,
                "rec_avg_loss": '{:.4f}'.format(rec_avg_loss / len(rec_cf_data_iter)),
                "joint_avg_loss": '{:.4f}'.format(joint_avg_loss / len(rec_cf_data_iter)),
                "cl_avg_loss": '{:.4f}'.format(cl_sum_avg_loss / (len(rec_cf_data_iter)*self.total_augmentaion_pairs)),
                "itemcl_loss": '{:.4f}'.format(itemcl_sum_avg_loss / (len(rec_cf_data_iter))),
            }
            for i, cl_individual_avg_loss in enumerate(cl_individual_avg_losses):
                post_fix['cl_pair_'+str(i)+'_loss'] = '{:.4f}'.format(cl_individual_avg_loss / len(rec_cf_data_iter))

            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix))

            with open(self.args.log_file, 'a') as f:
                f.write(str(post_fix) + '\n')
            
            self.writer.add_scalar(tag="loss/train rec loss", scalar_value=rec_avg_loss / len(rec_cf_data_iter), global_step = epoch)
            self.writer.add_scalar(tag="loss/train cl loss", scalar_value=cl_sum_avg_loss / (len(rec_cf_data_iter)*self.total_augmentaion_pairs), global_step = epoch)
            self.writer.add_scalar(tag="loss/train loss", scalar_value=rec_avg_loss / len(rec_cf_data_iter), global_step = epoch)
            self.writer.add_scalar(tag="loss/itemcl loss", scalar_value=itemcl_sum_avg_loss / len(rec_cf_data_iter), global_step = epoch)

            answer_class_list = self.args.item_freq_class[answer_list.squeeze()]
            head_tail = {
                "high_freq": np.argwhere(answer_class_list==2).squeeze(),
                "mid_freq": np.argwhere(answer_class_list==1).squeeze(),
                "low_freq": np.argwhere(answer_class_list==0).squeeze(),
            }
            for key in head_tail:
                indexes = head_tail[key]
                pred_logits_list_part = pred_logits_list[indexes]
                self.writer.add_scalar(tag=f"train_loss/{key}", scalar_value=np.mean(pred_logits_list_part), global_step = epoch)
                # 为了方便，暂时这里暂时写为横轴为epoch，后面需要看一下横轴为iteration的
        else:
            rec_data_iter = tqdm(enumerate(dataloader),
                                  desc="Recommendation EP_%s:%d" % (str_code, epoch),
                                  total=len(dataloader),
                                  bar_format="{l_bar}{r_bar}")
            self.model.eval()

            pred_list = None

            if full_sort:
                answer_list = None
                for i, batch in rec_data_iter:
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    batch = tuple(t.to(self.device) for t in batch)
                    # user_ids, input_ids, target_pos, target_neg = batch
                    user_ids, input_ids, target_pos, target_neg = batch
                    answers = target_pos
                    recommend_output = self.model.transformer_encoder(input_ids) # zzx: [bs, 64]


                    # recommendation results

                    rating_pred = self.predict_full(recommend_output) # zzx: [bs, item_num]

                    rating_pred = rating_pred.cpu().data.numpy().copy()
                    batch_user_index = user_ids.cpu().numpy()
                    rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0
                    # reference: https://stackoverflow.com/a/23734295, https://stackoverflow.com/a/20104162
                    # argpartition T: O(n)  argsort O(nlogn)
                    ind = np.argpartition(rating_pred, -20)[:, -20:]
                    arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                    batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                    if i == 0:
                        pred_list = batch_pred_list
                        answer_list = answers.cpu().data.numpy()
                    else:
                        pred_list = np.append(pred_list, batch_pred_list, axis=0)
                        answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)

                answer_class_list = self.args.item_freq_class[answer_list.squeeze()]
                head_tail = {
                    "high_freq": np.argwhere(answer_class_list==2).squeeze(),
                    "mid_freq": np.argwhere(answer_class_list==1).squeeze(),
                    "low_freq": np.argwhere(answer_class_list==0).squeeze(),
                }
                for key in head_tail:
                    indexes = head_tail[key]
                    answer_list_part, pred_list_part = answer_list[indexes], pred_list[indexes]
                    result_part = self.get_full_sort_score(epoch, answer_list_part, pred_list_part, name=key)
                    self.writer.add_scalar(tag=f"NDCG@20/{key}", scalar_value=result_part[0][5], global_step = epoch)
                
                result = self.get_full_sort_score(epoch, answer_list, pred_list, name="all")
                self.writer.add_scalar(tag=f"NDCG@20/all", scalar_value=result[0][5], global_step = epoch)
                return result    

            else:
                for i, batch in rec_data_iter:
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers, sample_negs = batch
                    recommend_output = self.model.finetune(input_ids)
                    test_neg_items = torch.cat((answers, sample_negs), -1)
                    recommend_output = recommend_output[:, -1, :]

                    test_logits = self.predict_sample(recommend_output, test_neg_items)
                    test_logits = test_logits.cpu().detach().numpy().copy()
                    if i == 0:
                        pred_list = test_logits
                    else:
                        pred_list = np.append(pred_list, test_logits, axis=0)

                return self.get_sample_scores(epoch, pred_list)
