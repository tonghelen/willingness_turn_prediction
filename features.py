import numpy as np
from utils import *


def getData(mods, task):
    '''
    Inputs:  
      mod     - list of modalities from 'X_speech', 'X_language' and 'X_vision',
                where 'X' is either 'spk' or 'lis'
      shuffle - whether to shuffle the data
    Returns: features and labels
      features:
        type  - np.array
        shape - (n,) each row is a list of features
        order - speaker's features, listener's features,
                each in order of speech, language, vision
      labels:
        type  - np.array
        shape - (n, 4)
        order - speakers'WoS, speakers'WoL, listeners'WoS, listeners'WoL
    '''
    print('Using modalities of', mods)
    print('Get labels of', task)
    labels = getLabel(task)
    
    features = []
    mods = set(mods)
    if 'spk_speech' in mods:
        features.append(getSpeechFeatures('spk'))
    if 'spk_language' in mods:
        features.append(getLanguageFeatures('spk'))
    if 'spk_vision' in mods:
        features.append(getVisionFeatures('spk'))
    if 'lis_speech' in mods:
        features.append(getSpeechFeatures('lis'))
    if 'lis_language' in mods:
        features.append(getLanguageFeatures('lis'))
    if 'lis_vision' in mods:
        features.append(getVisionFeatures('lis'))
        
    new_features = np.array([[features[i][j] for i in range(len(mods))] \
                             for j in range(labels.shape[0])]) 
    
    return new_features, labels


def getLabel(task):
    '''
    Returns: Labels
        type  - np.array
        shape - (n, 4) where n=2861
        order - speakers'WoS, speakers'WoL, listeners'WoS, listeners'WoL, next_spk
    '''
    labels = loadPickle(DATADIR+'labels_cls.pkl')
    if task == 'reg':
        return labels[:,:-1]
    elif task == 'cls':
        return labels[:,-1][:,None]
    else:
        return labels


def getSpeechFeatures(who):
    '''This function gets vggish features
    Returns: features
        type  - np.array
        shape - (n,) each row is (nframes, 128)
    '''
    features = loadPickle(DATADIR+f'{who}_speech_feat.pkl')
    return features
    

def getLanguageFeatures(who):
    '''This function gets bert features trained on Japanese wikipedia
    Returns: features
        type  - np.array
        shape - (n, d) which is (2861,768)
    '''
    features = loadPickle(DATADIR+f'{who}_language_feat.pkl')
    return features


def getVisionFeatures(who):
    '''This function gets resnet50 features trained on ImageNet
    Returns: features
        type  - np.array
        shape - (n,) each row is (nframes, 2048)
    '''
    features = loadPickle(DATADIR+f'{who}_vision_feat.pkl')
    return features



if __name__ == '__main__':
    X, y = getData(['spk_vision','spk_speech','spk_language',
                   'lis_vision','lis_speech','lis_language'], 'cls')
    print('Features shape:', X.shape)
    print('Labels shape:', y.shape)
