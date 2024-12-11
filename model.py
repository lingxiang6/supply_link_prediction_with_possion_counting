import torch
from pathlib import Path
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import warnings
import copy
import os
import json
import sys
import shutil
from tqdm import tqdm
import pandas as pd
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from NHPE import lambda_16_predict
from NHPE import * 
import pickle
from sklearn import metrics

# 使用argparse处理命令行参数
parser = argparse.ArgumentParser(description='GNN Link Prediction Model Training')
parser.add_argument('--dataset', type=str, default='supplychain', help='dataset name')
parser.add_argument('--model_name', type=str, default='gnn_link_prediction', help='model name')
parser.add_argument('--embedding_name', type=str, default='qwen', help='embedding name [qwen, bert]')
parser.add_argument('--feature', type=str, default='qwen', help='embedding method,[qwen, bert]')
parser.add_argument('--embedding_path_qwen', type=str, default='/your/root', help='path to embedding file')
parser.add_argument('--embedding_path_bert', type=str, default='/your/root', help='path to embedding file')
parser.add_argument('--edge_status_json', type=str, default='/your/root/edge_status.json', help='path to edge_status.json')
parser.add_argument('--adj17_path', type=str, default='/your/root', help='path to adjacency matrix file')
parser.add_argument('--a15_path', type=str, default='/your/root', help='path to adjacency matrix file')
parser.add_argument('--industry_path', type=str, default='/your/root', help='path to adjacencyindustry information')
parser.add_argument('--node_index', type=str, default='/your/root', help='path to node_index')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--cuda', type=int, default=0, help='CUDA device ID')
parser.add_argument('--alpha', type=float, default=0.5, help='parameter alpha')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--batch_size', type=int, default=5000, help='batch size')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
parser.add_argument('--hidden_channels', type=int, default=512, help='number of hidden channels[512, 256]')
parser.add_argument('--out_channels', type=int, default=128, help='number of output channels')
parser.add_argument('--use_location', default=True, help='whether to use location information')
parser.add_argument('--location_path_qwen', type=str, default='you/root', help='path to location file')
parser.add_argument('--location_path_bert', type=str, default='you/root', help='path to location file numpy')
parser.add_argument('--smothness', default=True, help='whether to use smothness')
parser.add_argument('--industry', default=True, help='whether to use industry')

args = parser.parse_args()

# 设置设备
args.device = f'cuda:{args.cuda}' if torch.cuda.is_available() and args.cuda >= 0 else 'cpu'
# 更新模型名称
args.model_name = f'{args.model_name}_{args.embedding_name}'
print("参数设置:")
for key, value in vars(args).items():
    print(f"  {key}: {value}")

def set_random_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

class EdgeDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        self.edge_label = data.edge_label
        self.edge_label_index = data.edge_label_index.t()
        self.a16_hat = lambda_16_predict

    def __len__(self):
        return self.edge_label_index.size(0)

    def __getitem__(self, idx):
        edge = self.edge_label_index[idx,:]
        label = self.edge_label[idx]
        edge_str = f"{edge[0].item()}_{edge[1].item()}"
        try:
            predicted_value = self.a16_hat.loc[edge_str, 'predicted_value']
        except KeyError:
            predicted_value = 0.0  
        predicted_value=torch.from_numpy(np.array(predicted_value)).to(torch.float32)
        return edge, label, predicted_value

def get_edge_data_loader(train_data, val_data, test_data, batch_size):
    train_dataset = EdgeDataset(train_data)
    val_dataset = EdgeDataset(val_data)
    test_dataset = EdgeDataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,drop_last=True)    
    return train_loader, val_loader, test_loader

class GCNLinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, args):
        super(GCNLinkPredictor, self).__init__()
        self.linear= torch.nn.Linear(in_channels,args.hidden_channels)
        self.conv1 = GCNConv(args.hidden_channels, args.hidden_channels)
        self.conv2 = GCNConv(args.hidden_channels, args.hidden_channels)
        self.dropout = torch.nn.Dropout(0.5)
        self.bns = torch.nn.BatchNorm1d(in_channels)
        self.smothness = args.smothness
        self.industry = args.industry
        self.lamda_15_1 = nn.Parameter(torch.randn(args.batch_size, 1), requires_grad=True)
        self.alpha = args.alpha
        self.linear_location = nn.Linear(in_channels, 512)
        self.location_vector_qwen = torch.tensor(np.load(args.location_path_qwen), dtype=torch.float32).to(args.device)
        self.location_vector_bert = torch.tensor(np.load(args.location_path_bert), dtype=torch.float32).to(args.device)
        self.use_location = args.use_location
        self.adj15 = self.supply_industry_fusion(args)
        
    def matrix_process(self, adj1):
        supply_edges = adj1
        demand_edges = adj1[[1, 0],:]
        return supply_edges, demand_edges
    
    def distance(self, x, edge_index):
            index = edge_index.t()
            x1 = x[index[0, :]]
            x2 = x[index[1, :]]
            cos_sim = F.cosine_similarity(x1, x2, dim=1)
            return cos_sim
    
    def supply_industry_fusion(self, args):
        with open(args.industry_path, 'rb') as file:
            industry_matrix = pickle.load(file)
        adj15 = pd.read_csv(args.a15_path).iloc[:, 1:] 
        adj15 = adj15[adj15.apply(lambda row: row['customer'] in industry_matrix.get(row['supplier'], set()), axis=1)]
        with open(args.node_index, 'r') as f:
            company_dict = json.load(f)
        adj15['supplier'] = adj15['supplier'].map(company_dict)
        adj15['customer'] = adj15['customer'].map(company_dict)
        adj15 = torch.tensor(adj15.values).to(device=args.device)
        return adj15
      
    def forward(self, x, edge_index, lambda_16_hat):
        if self.industry:
            supply_edges,  demand_edges = self.matrix_process(self.adj15)
        else:
            supply_edges,  demand_edges = self.matrix_process(edge_index)
        supply_edges,  demand_edges = supply_edges.t(),  demand_edges.t()###转置
        x = self.bns(x)
        x = self.linear(x)
        s = F.leaky_relu(self.conv1(x, supply_edges))  
        s = self.dropout(s)
        s = F.leaky_relu(self.conv2(s, supply_edges)) 
        r = F.leaky_relu(self.conv1(x, demand_edges))  
        r = self.dropout(r)
        r = F.leaky_relu(self.conv2(r, demand_edges)) 
        
        index = edge_index.t()
        s1 = s[index[0,:]]
        r1 = r[index[1,:]]
        # logits=torch.mul(s1, r1).sum(dim=1) / (torch.norm(s1, dim=1) * torch.norm(r1, dim=1))
        logits=torch.mul(s1, r1).sum(dim=1)
        if self.use_location:
            if args.feature == 'qwen':
                location = self.linear_location(self.location_vector_qwen)
            else:
                location = self.linear_location(self.location_vector_bert)
            cos=self.distance(location, edge_index)
            logits /=cos
        
        if self.smothness:
            lamda_17 = self.get_lamda_17(logits, lambda_16_hat)
            return lamda_17
        return logits
    
    def get_lamda_17(self, x, lambda_16_hat):
        lambda16_hat_1 = self.alpha * lambda_16_hat + (1-self.alpha) * self.lamda_15_1
        lambda16_hat_2 = self.alpha * lambda16_hat_1 + (1-self.alpha) * x.unsqueeze(1)
        b16 = 2 * lambda16_hat_1 - lambda16_hat_2 
        c16 = self.alpha / (1-self.alpha) * (lambda16_hat_1 - lambda16_hat_2)
        lamda_17 = b16 + c16
        return lamda_17.squeeze()

def read_data(args):
    adj17 = pd.read_csv(args.adj17_path)
    adj17 = torch.tensor(adj17.values).transpose(1,0).to(device=args.device)    
    transform = RandomLinkSplit(
    num_val=0.1, num_test=0.2, is_undirected=False,
    neg_sampling_ratio=1.0)
    if args.feature == 'qwen':
        company_feature = torch.tensor(np.load(args.embedding_path_qwen), dtype=torch.float32)
    else:
        company_feature = torch.tensor(np.load(args.embedding_path_bert), dtype=torch.float32)
    data = Data(x=company_feature, edge_index=adj17).to(device=args.device)
    train_data, val_data, test_data = transform(data)
    return data, train_data, val_data, test_data

def val(loder, model, args):
    model.eval()
    with torch.no_grad():
        y_trues = []
        y_predicts = []
        total_loss=0.0
        for batch, (edge_index, edge_label,lambda_16_hat)in enumerate(loder):
            lambda_16_hat = torch.tensor(lambda_16_hat)
            lambda_16_hat = lambda_16_hat.clone().detach().unsqueeze(1).to(args.device)
            score = model(data.x, edge_index, lambda_16_hat)
            loss = F.binary_cross_entropy_with_logits(score, edge_label)
            y_trues.append(edge_label.detach().cpu())
            y_predicts.append(score.detach().cpu())
            total_loss += loss.item()
        total_loss /= (batch + 1)
        y_trues = torch.cat(y_trues, dim=0)
        y_predicts = torch.cat(y_predicts, dim=0)
        
        predicted_classes = (y_predicts >= 0.5).float()
        
        correct_predictions = (predicted_classes == y_trues).float()
        accuracy = correct_predictions.mean().item()  # 计算平均准确率
        return accuracy, total_loss

def test(loder, model, args):
    model.eval()
    with torch.no_grad():
        y_trues = []
        y_predicts = []
        total_loss=0.0
        for batch, (edge_index, edge_label,lambda_16_hat)in enumerate(loder):
            lambda_16_hat = torch.tensor(lambda_16_hat)
            lambda_16_hat = lambda_16_hat.clone().detach().unsqueeze(1).to(args.device)
            score = model(data.x, edge_index, lambda_16_hat)
            loss = F.binary_cross_entropy_with_logits(score, edge_label)
            y_trues.append(edge_label.detach().cpu())
            y_predicts.append(score.detach().cpu())
            total_loss += loss.item()
        total_loss /= (batch + 1)
        y_trues = torch.cat(y_trues, dim=0)
        y_predicts = torch.cat(y_predicts, dim=0)
        
        predicted_classes = (y_predicts >= 0.5).float()
        # 计算 Precision:TP,表示实际为正例，预测也为正例；FP，表示实际为负例，预测为正例
        TP = ((y_trues == 1) & (predicted_classes == 1)).sum().float()
        FP = ((y_trues == 0) & (predicted_classes == 1)).sum().float()
        precision = TP / (TP + FP) ##if (TP + FP) > 0 else torch.tensor(0.0)


        # 计算 Recall：FN，实际为正例，但是预测为负例
        FN = ((y_trues == 1) & (predicted_classes == 0)).sum().float()
        recall = TP / (TP + FN) if (TP + FN) > 0 else torch.tensor(0.0)


        # 计算 F1 score
        f1 = 2 * (precision * recall) / (precision + recall)   #if (precision + recall) > 0 else torch.tensor(0.0)

        
        ##计算auc
        auc = metrics.roc_auc_score(y_trues.numpy(), y_predicts.numpy())
        correct_predictions = (predicted_classes == y_trues).float()
        accuracy = correct_predictions.mean().item()  # 计算平均准确率
        return accuracy, total_loss, precision, recall, f1, auc 
        
if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")
    set_random_seed(args.seed)
    print(f'loading dataset {args.dataset}...')
    data, train_data, val_data, test_data = read_data(args)
    print(f'get edge data loader...')
    train_loader, val_loader, test_loader = get_edge_data_loader(
                                                                 train_data=train_data,
                                                                 val_data=val_data,
                                                                 test_data=test_data,
                                                                 batch_size=args.batch_size)
    model = GCNLinkPredictor(in_channels=data.x.shape[1], args=args).to(device=args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    train_steps = 0
    for epoch in range(args.epochs):
        model.train()

        train_y_trues = []
        train_y_predicts = []
        train_total_loss = 0.0
        train_loader_tqdm = tqdm(train_loader, ncols=120)
        for batch, (edge_index, edge_label, lambda_16_hat)in enumerate(train_loader_tqdm):

            lambda_16_hat = torch.tensor(lambda_16_hat)
            lambda_16_hat = lambda_16_hat.clone().detach().unsqueeze(1).to(args.device)
            
            score = model(data.x, edge_index, lambda_16_hat)
            loss = F.binary_cross_entropy_with_logits(score, edge_label)

            train_total_loss += loss.item()
            
            train_y_trues.append(edge_label.detach().cpu())
            train_y_predicts.append(score.detach().cpu())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_steps += 1

        train_total_loss /= (batch + 1)
        train_y_trues = torch.cat(train_y_trues, dim=0)
        train_y_predicts = torch.cat(train_y_predicts, dim=0)
        
        predicted_classes = (train_y_predicts >= 0.5).float()
        
        ###计算acc
        correct_predictions = (predicted_classes == train_y_trues).float()
        accuracy = correct_predictions.mean().item()  # 计算平均准确率
        
        #验证集
        vli_accuracy,loss = val(val_loader, model, args)
        
        print(f'Epoch: {epoch}, train loss: {train_total_loss:.4f}, train accuracy: {accuracy:.4f}', 
      f'vali loss: {loss:.4f}, vali accuracy: {vli_accuracy:.4f}')
    test_accuracy,loss, test_precision, test_recall, test_f1, test_auc = test(test_loader, model, args)
    print(f'test accuracy: {test_accuracy:.4f}, test loss: {loss:.4f}, test_precision: {test_precision:.4f}, test_recall:{test_recall:.4f}, test_f1:{test_f1:.4f}, test_auc:{test_auc:.4f}')





