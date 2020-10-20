import argparse
import numpy as np
import os

from m3 import M3
from features import getData
from utils import *


def parse_args():
    parser = argparse.ArgumentParser(description='^-^')
    parser.add_argument('--savedir', '-dir', type=str, default='tmp')
    
    parser.add_argument('--model', '-a', type=str, default='m3')
    parser.add_argument('--mods', '-m', type=str, default= \
                        'spk_vision,spk_speech,spk_language,lis_vision,lis_speech,lis_language',
                        help='choose from spk/lis_speech/language/vision')
    parser.add_argument('--label', '-L', type=int, default=4,
                        choices=[0,1,2,3,4], help='which score to predict, 4 represents all')
    parser.add_argument('--task', '-t', type=str, default='cls',
                        choices=['cls', 'reg', 'multi'])
    
    parser.add_argument('--epoch', '-e', type=int, default=50)
    parser.add_argument('--batch', '-b', type=int, default=64)
    parser.add_argument('--dropout', '-d', type=float, default=0.1)
    parser.add_argument('--lr', '-lr', type=float, default=0.0001)
    parser.add_argument('--in_dims', '-in', type=int, default=64)
    parser.add_argument('--out_dims', '-out', type=int, default=192)
    
    args = parser.parse_args()
    return args


def cross_validation(args, features, labels):
    log = os.path.join(args.path, 'log.txt')
    text = f'Features: {args.mods}\nTask: {args.task}\n'
    text += f'Epochs: {args.epoch}, Batch: {args.batch}, Dropout: {args.dropout}\n'
    text += f'LR: {args.lr}, In_dims: {args.in_dims}, Out_dims: {args.out_dims}\n\n'
    with open(log, 'w') as f:
        f.write(text)

    model_creator = M3 if args.model == 'm3' else None
    loss = []
    for i, (train_idx, test_idx) in enumerate(data_split()):
        print(f'------Round {i}------')
        
        train_data = (features[train_idx], labels[train_idx])
        test_data = (features[test_idx], labels[test_idx])
        
        model = model_creator(
            args.mods, args.task,
            args.lr, args.in_dims, args.out_dims, labels.shape[1], args.dropout)
        train_loss, test_loss = model.train(
            train_data, test_data, args.epoch, args.batch,
            os.path.join(RESULTSDIR, args.savedir, 'log', str(i))
        )
        pred = model.predict(test_data[0])
        
        save_result(args, i, train_loss, test_loss, model, pred)
        loss.append(test_loss)
        print('Test loss', test_loss)
        
    print('\nMean(std): ', np.mean(loss), np.std(loss))
    with open(log, 'a') as f:
        f.write('Overall results: %.4f(%.4f)'%(np.mean(loss), np.std(loss)))

    
    
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


def save_result(args, num, train_loss, test_loss, model, pred):
    log = os.path.join(args.path, 'log.txt')
    text = '%2d\t%.4f\t%.4f\n' % (num, train_loss, test_loss)
    with open(log, 'a') as f:
        f.write(text)
       
    mdl = os.path.join(args.path, 'model/model_%d'%num)
    model.save(mdl)
    np.save(args.path+'/predict/pred_%d'%num, pred)
    


if __name__ == '__main__':
    args = parse_args()
    args.mods = args.mods.split(',')

    path = os.path.join(RESULTSDIR, args.savedir)
    args.path = path
    if not os.path.exists(path):
        os.mkdir(path)
        os.mkdir(path+'/model')
        os.mkdir(path+'/log')
        os.mkdir(path+'/predict')

    features, labels = getData(args.mods, args.task)

    if args.task == 'reg':
        if args.label < 4:
            cross_validation(args, features, labels[:,args.label][:,None]) # only test 1 score
        else:
            cross_validation(args, features, labels) # test 4 scores
    else:
        cross_validation(args, features, labels)
