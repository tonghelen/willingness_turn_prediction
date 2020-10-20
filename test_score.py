import argparse
import numpy as np
import os
import scipy.stats as stats
from sklearn.metrics import f1_score

import torch
import torch.nn as nn

from features import getLabel
from utils import *


def parse_args():
    parser = argparse.ArgumentParser(description='^-^')
    parser.add_argument('--task', '-t', type=str, default='cls',
                        choices=['cls', 'reg', 'multi'])
    parser.add_argument('--cor', type=str, default='spearman',
                        choices=['pearson', 'spearman'])
    parser.add_argument('--label', '-L', type=int, default=4,
                        choices=[0,1,2,3,4], help='which score to predict, 4 represents all')
    parser.add_argument('--data', '-d', type=str, default='test')
    parser.add_argument('--dir', '-dir', type=str, default='predict')
    args = parser.parse_args()
    return args


def cross_validation(args, labels):
    if args.task == 'reg' or args.task == 'multi':
        criterion = nn.MSELoss(reduction='mean')
    
    scores, losses, f1s = [], [], []
    for i, (train_idx, test_idx) in enumerate(data_split()):
#         print(f'------Round {i}------')
        y = labels[train_idx] if args.data == 'train' else labels[test_idx]
        pred = np.load('%s/pred_%d.npy' % (args.dir, i))
        
        if args.task == 'reg':
            score, loss = [], []
            for j in range(y.shape[1]):
                if args.cor == 'pearson':
                    score.append(stats.pearsonr(pred[:,j], y[:,j])[0])
                else:
                    score.append(stats.spearmanr(pred[:,j], y[:,j])[0])
                loss.append(criterion(torch.FloatTensor(pred[:,j]),
                                      torch.FloatTensor(y[:,j])).item())
            scores.append(score)
            losses.append(loss) 
#             print('Test score', score)
#             print('Test loss', loss)
            
        elif args.task == 'cls':
            pred_y = np.argmax(pred, axis=1)
            f1 = list(f1_score(y.reshape(-1), pred_y, average=None))
            f1.append(f1_score(y.reshape(-1), pred_y, average='macro'))
            f1.append(f1_score(y.reshape(-1), pred_y, average='weighted'))
            f1s.append(f1)
#             print('Test score', f1)

        else:
            y1 = y[:,:-1]
            y2 = y[:,-1]
            score, loss = [], []
            for j in range(y1.shape[1]):
                if args.cor == 'pearson':
                    score.append(stats.pearsonr(pred[:,j], y[:,j])[0])
                else:
                    score.append(stats.spearmanr(pred[:,j], y1[:,j])[0])
                loss.append(criterion(torch.FloatTensor(pred[:,j]),
                                      torch.FloatTensor(y1[:,j])).item())
            scores.append(score)
            losses.append(loss)
            pred_y = np.argmax(pred[:,-2:], axis=1)
            f1 = list(f1_score(y2, pred_y, average=None))
            f1.append(f1_score(y2, pred_y, average='macro'))
            f1.append(f1_score(y2, pred_y, average='weighted'))
            f1s.append(f1)
    
#    print('\n')
    if args.task == 'reg' or args.task == 'multi':
        mean_score = np.mean(scores, axis=0)
        std_score = np.std(scores, axis=0)
        mean_loss = np.mean(losses, axis=0)
        std_loss = np.std(losses, axis=0)
        print("Corr:")
        print('%.4f(%.4f)\t%.4f(%.4f)\t%.4f(%.4f)\t%.4f(%.4f)' % (
            mean_score[0],std_score[0],
            mean_score[1],std_score[1],
            mean_score[2],std_score[2],
            mean_score[3],std_score[3]))
        print("MSE:")
        print('%.4f(%.4f)\t%.4f(%.4f)\t%.4f(%.4f)\t%.4f(%.4f)' % (
            mean_loss[0],std_loss[0],
            mean_loss[1],std_loss[1],
            mean_loss[2],std_loss[2],
            mean_loss[3],std_loss[3]))

    if args.task == 'cls' or args.task == 'multi':
        mean_score = np.mean(f1s, axis=0)
        std_score = np.std(f1s, axis=0)
        print("F1:")
        print('%.4f(%.4f)\t%.4f(%.4f)\t%.4f(%.4f)\t%.4f(%.4f)' % (
            mean_score[0],std_score[0],
            mean_score[1],std_score[1],
            mean_score[2],std_score[2],
            mean_score[3],std_score[3]))
        print(' & %.3f(%.3f) & %.3f(%.3f) & %.3f(%.3f)' % (
            mean_score[0],std_score[0],
            mean_score[1],std_score[1],
            mean_score[3],std_score[3]))
    
    
def data_split():
    group_dict = loadPickle(DATADIR+'group_dict_cls.pkl')
    keys = list(group_dict.keys())
    
    n = len(group_dict)
    tr_idx = np.arange(n-1)
    te_idx = n-1
    
    for i in range(n):
        train_idx = [group_dict[keys[idx]] for idx in tr_idx]
        train_idx = np.concatenate(train_idx)
        test_idx = group_dict[keys[te_idx]]
        yield train_idx, test_idx
        tr_idx = (tr_idx+1) % n
        te_idx = (te_idx+1) % n    



if __name__ == '__main__':
    args = parse_args()
    labels = getLabel(args.task)
    
    if args.task == 'reg':
        if args.label < 4:
            cross_validation(args, labels[:,args.label][:,None]) # only test 1 score
        else:
            cross_validation(args, labels) # test 4 scores
    else:
        cross_validation(args, labels)
