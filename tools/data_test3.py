import numpy as np


if __name__ == '__main__':

    # file name
    file_name = './matrix.npy'

    # write
    num = 100000
    data = np.random.rand(num, num)
    fp = np.memmap(file_name, dtype='float32', mode='w+', shape=(num, num))
    fp[:] = data[:]
    fp.flush()
    print("End!")

    # read
    newfp = np.memmap(file_name, dtype='float32', mode='r')
    print(f'old shape: {newfp.shape}   old type: {newfp.dtype}')
    length = int(np.sqrt(newfp.shape[0]))
    score_matrix = newfp.reshape(length, length)
    print(f'new shape: {score_matrix.shape}   new type: {score_matrix.dtype}')
    print(score_matrix[9999][9999])
    print(score_matrix[9999, 9999])
    print(score_matrix[9999][9999].dtype)
