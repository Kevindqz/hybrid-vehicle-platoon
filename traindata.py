import pickle
# load pickle file and return the data
def load_data(file):
    with open(file, 'rb') as f:
        train_data = pickle.load(f)
    return train_data

if __name__ == '__main__':
    file = 'train_data.pkl'
    data = load_data(file)
    # only print the keys of the dictionary
    print(data.keys())
