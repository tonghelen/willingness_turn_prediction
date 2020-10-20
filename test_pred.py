import argparse
import numpy as np
import os

from m3 import M3
from features import getData
from utils import *


def parse_args():
    parser = argparse.ArgumentParser(description='^-^')
    parser.add_argument('--savedir', '-dir', type=str, default='tmp')
    
    parser.add_argument('--model', '-a', type=str, default='m3',
                        choices=['svr', 'm3'])
    parser.add_argument('--mods', '-m', type=str, default= \
                        'spk_vision,spk_speech,spk_language,lis_vision,lis_speech,lis_language',
                        help='choose from spk/lis_speech/language/vision')
    parser.add_argument('--label', '-L', type=int, default=4,
                        choices=[0,1,2,3,4], help='which score to predict, 4 represents all')
    
    parser.add_argument('--dropout', '-d', type=float, default=0.5)
    parser.add_argument('--lr', '-lr', type=float, default=0.001)
    parser.add_argument('--in_dims', '-in', type=int, default=64)
    parser.add_argument('--out_dims', '-out', type=int, default=192)
    
    args = parser.parse_args()
    return args


def cross_validation(args, features, labels):
    model_creator = M3 if args.model == 'm3' else None
    path = os.path.join(RESULTSDIR, args.savedir)
    if not os.path.exists(path+'/predict/'):
        os.mkdir(path+'/predict/')
    for i, (train_idx, test_idx) in enumerate(data_split()):
        print(f'------Round {i}------')
       
        train_data = (features[train_idx], labels[train_idx])
        test_data = (features[test_idx], labels[test_idx])
        
        mdl = os.path.join(path, 'model/model_%d' % i)
        model = model_creator(args.mods, args.lr, args.in_dims, args.out_dims,
                              labels.shape[1], args.dropout)
        model.load(mdl)
        
        pred = model.predict(train_data[0])
        	
        np.save(path+'/predict/pred_%d' % i, pred)
    
    
def data_split():
    group_dict = loadPickle(DATADIR+'group_dict.pkl')
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
    args.mods = args.mods.split(',')

    features, labels = getData(args.mods)

    if args.label < 4:
        cross_validation(args, features, labels[:,args.label][:,None]) # only test 1 score
    else:
        cross_validation(args, features, labels) # test 4 scores
