import numpy as np
#import scipy.stats as stats

import torch
import torch.nn as nn
import torch.optim as optim


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("GPU available")


class RNN(nn.Module):
    def __init__(self, in_size, h_size, n_layers, dropout=0):
        super(RNN, self).__init__()
        self.model = nn.GRU(in_size,hidden_size=h_size,num_layers=n_layers, 
                            batch_first=True,dropout=dropout)
    def forward(self, x):
        out, _ = self.model(x, None)
        return out[0][-1][None] #(1,h_size)


class MultiModel(nn.Module):
    def __init__(self, mods, in_dims, out_dims, n_scores, dp=0.1):
        super(MultiModel, self).__init__()
        
        layers = []
        mods = set(mods)
        if 'spk_speech' in mods:
            layers.append(RNN(128, in_dims, 1))
        if 'spk_language' in mods:
            layers.append(nn.Linear(768, in_dims))
        if 'spk_vision' in mods:
            layers.append(RNN(2048, in_dims, 1))
        if 'lis_speech' in mods:
            layers.append(RNN(128, in_dims, 1))
        if 'lis_language' in mods:
            layers.append(nn.Linear(768, in_dims))
        if 'lis_vision' in mods:
            layers.append(RNN(2048, in_dims, 1))
            
        self.layers = nn.ModuleList(layers)
        self.dp1 = nn.Dropout(dp, inplace=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(in_dims*len(mods), out_dims)
        self.dp2 = nn.Dropout(dp, inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(out_dims, n_scores)
        
    def forward(self, data):
        feats = []
        for layer, x in zip(self.layers, data):
            feats.append(layer(x[None])) # (1, in_dims)
        y = torch.cat(feats, dim=1) # (1, in_dims * len(mods))
        
        y = self.dp1(y)
        y = self.relu1(y)
        y = self.fc1(y)
        y = self.dp2(y)
        y = self.relu2(y)
        y = self.fc2(y)
        
        return y #(1, n_scores)
    
    
class M3:
    def __init__(self, mods, task, lr, in_dims, out_dims, n_scores, dp):
        print('Using multi-modalities model, hyper-params:', lr, in_dims, out_dims)
        self.mods = mods
        self.task = task
        
        if task == 'reg':
            self.model = MultiModel(mods, in_dims, out_dims, n_scores, dp).to(device)
            self.criterion = nn.MSELoss(reduction='sum')
        elif task == 'cls':
            self.model = MultiModel(mods, in_dims, out_dims, n_scores+1, dp).to(device)
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.model = MultiModel(mods, in_dims, out_dims, n_scores+1, dp).to(device)
            self.criterion_reg = nn.MSELoss(reduction='sum')
            self.criterion_cls = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        #self.optimizer = optim.SGD(self.model.parameters(), lr=lr)    

    def test(self, data):
        self.model.eval()
        X, Y = data

        loss = 0
        for x, y in zip(X, Y):
            x = [torch.FloatTensor(i).to(device) for i in x]
            with torch.no_grad():
                if self.task == 'reg':
                    y = torch.FloatTensor(y).to(device)
                    pred = self.model(x).reshape(y.shape)
                    loss += self.criterion(pred, y).item()  
                elif self.task == 'cls':
                    y = torch.LongTensor(y).to(device)
                    pred = self.model(x)
                    loss += self.criterion(pred, y).item() 
                else:
                    y1 = torch.FloatTensor(y[:-1]).to(device)
                    y2 = torch.LongTensor(y[-1][None]).to(device)
                    pred = self.model(x)
                    pred1 = pred[:,:-2].reshape(y1.shape)
                    pred2 = pred[:,-2:]
                    loss1 = self.criterion_reg(pred1, y1).item()
                    loss2 = self.criterion_cls(pred2, y2).item()
                    loss += (loss1/9 + loss2*8/9)
        return loss/Y.shape[0]
    
    def train(self, train_data, eval_data, epoch, batch, path):
        X, y = train_data
        n = y.shape[0]
        #y = y + (np.random.random(y.shape)-0.5)*0.5
        
        for e in range(epoch):
            self.model.train()
            idx = np.arange(n)
            np.random.shuffle(idx)
            tr_X, tr_y = X[idx,:], y[idx,:]
            tr_loss = .0
            
            loss = []
            for b, (b_x, b_y) in enumerate(zip(tr_X, tr_y)):
                b_x = [torch.FloatTensor(i).to(device) for i in b_x]
                
                if self.task == 'reg':
                    b_y = torch.FloatTensor(b_y).to(device)
                    pred = self.model(b_x).reshape(b_y.shape)
                    loss.append(self.criterion(pred, b_y))
                elif self.task == 'cls':
                    b_y = torch.LongTensor(b_y).to(device)
                    pred = self.model(b_x)
                    loss.append(self.criterion(pred, b_y))
                else:
                    b_y1 = torch.FloatTensor(b_y[:-1]).to(device)
                    b_y2 = torch.LongTensor(b_y[-1][None]).to(device)
                    pred = self.model(b_x)
                    pred1 = pred[:,:-2].reshape(b_y1.shape)
                    pred2 = pred[:,-2:]
                    loss1 = self.criterion_reg(pred1, b_y1)
                    loss2 = self.criterion_cls(pred2, b_y2)
                    loss.append(loss1/9 + loss2*8/9)
                    
                tr_loss += loss[-1].item()
                
                if (b+1) % batch == 0:
                    self.optimizer.zero_grad()
                    batch_loss = sum(loss)/len(loss)
                    batch_loss.backward()
                    self.optimizer.step()
                    loss = []
            if n % batch:
                self.optimizer.zero_grad()
                batch_loss = sum(loss)/len(loss)
                batch_loss.backward()
                self.optimizer.step()
                
            with open(path+'_train','a') as f:
                f.write('%.6f\n' % (tr_loss/n))
                
            te_loss = self.test(eval_data)
            with open(path+'_test','a') as f:
                f.write('%.6f\n' % te_loss)
            print('Epoch: %d\tTrain Loss: %.6f\tEval Loss: %.6f' % (e, tr_loss/n, te_loss))

            if (e+1) % 25 == 0: # LR decay by 0.1
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= 0.1
                        
        tr_loss = self.test(train_data)
        te_loss = self.test(eval_data)
        return tr_loss, te_loss
      
    def predict(self, X):
        self.model.eval()

        pred = []
        for x in X:
            x = [torch.FloatTensor(i).to(device) for i in x]
            with torch.no_grad():
                pred.append(self.model(x).detach().cpu().numpy())
        pred = np.concatenate(pred, axis=0) # (n, n_scores)
        return pred
    
    def save(self, file):
        model = {'mods': self.mods,
                 'task': self.task,
                 'state_dict': self.model.cpu().state_dict(),
                 'optimizer' : self.optimizer.state_dict(),
                }
        torch.save(model, file)

    def load(self, file):
        model = torch.load(file)
        self.model.load_state_dict(model['state_dict'])
        self.optimizer.load_state_dict(model['optimizer'])
