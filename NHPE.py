import json
import pandas as pd
from model import args
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.init as init
# import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
device = 'cuda:1'


with open(args.edge_status_json, 'r') as f:
    edge_status = json.load(f)

dataset = pd.DataFrame.from_dict(edge_status, orient='index')

def poisson_counting_process(dataset, style = None):
    data = dataset.apply(lambda x: (x == 1).cumsum(), axis=1)
    return data
# # def poisson_exp(data_exp):
# #     data_exp.iloc[:, :4] = np.exp(data_exp.iloc[:, :4])
# #     return data_exp
    
data = poisson_counting_process(dataset)
scaler = StandardScaler()
data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)


X = data.iloc[:, :15].values
y = data.iloc[:, 16].values



X_tensor = torch.tensor(X, dtype = torch.float32).to(device)
y_tensor = torch.tensor(y, dtype = torch.float32).to(device)


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(15, 1)
        init.uniform_(self.linear.weight, a=0.0, b=1.0)
        init.uniform_(self.linear.bias, a=0.0, b=1.0)
        
        
    def forward(self, x):
        x = self.linear(x)
        # x = F.softmax(x, dim=1) 
        return x


model = LinearRegressionModel().to(device)





criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01) 


epochs = 1000
for epoch in range(epochs):
 
    y_pred = model(X_tensor).squeeze()
    loss_predict = criterion(y_pred, y_tensor)
    optimizer.zero_grad() 
    loss_predict.backward()        
    optimizer.step()      
    
    if (epoch + 1) % 10 == 0:  
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss_predict.item():.4f}')



with torch.no_grad():
    y_test = model(X_tensor).cpu().numpy().flatten()  


data['predicted_value'] = y_test
data.reset_index(inplace=True)
lambda_16_predict = data






    
    