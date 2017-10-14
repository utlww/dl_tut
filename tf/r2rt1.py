import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

num_steps = 5 # truncated steps
batch_size = 200
num_classes = 2
state_size = 4
learning_rate = 0.1

def gen_data(size=1000000):
    X = np.array(np.random.choice(2, size=size))
    Y = []
    threshold = 0.5
    for i in range(size):
        if X[i-3] == 1:
            threshold += 0.5
        if X[i-8] == 1:
            threshold -= 0.25
        if np.random.rand() > threshold:
            Y.append(0)
        else:
            Y.append(1)
    return X, np.array(Y)


def get_batch(raw_data, batch_size, num_steps):
    raw_x, raw_y = raw_data
    data_length = len(raw_x)

    batch_partition_length =

def main():
    print(gen_data(10))


if __name__ == "__main__":
    main()
