import pickle


WORKPLACE = '../'
DATADIR = WORKPLACE + 'Features/'
RESULTSDIR = WORKPLACE + 'Results/'


def loadPickle(file):
    with open (file, 'rb') as f:
        data = pickle.load(f)
    return data

def savePickle(file, data):
    with open (file, 'wb') as f:
        pickle.dump(data, f)