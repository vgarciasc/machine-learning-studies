import pickle

def save_model(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def load_model(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)