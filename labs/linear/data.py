from core import Dataset

def read_data(data_path):
    file = open(data_path, "r")
    m = int(file.readline())

    train = __read_dataset__(file, m)
    test = __read_dataset__(file, m)

    return train, test
    

def __read_dataset__(input_file, m):
    n = int(input_file.readline())

    X = []
    Y = []

    for _ in range(n):
        arr = [int(v) for v in input_file.readline().split()]

        curr_x = arr[:m]
        curr_x.append(1)
        X.append(curr_x)
        Y.append(arr[m])
    
    return Dataset(X, Y, m + 1)

