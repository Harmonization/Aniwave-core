import numpy as np

def PPI(hsi, q, thr=1, n_skewers=10000):
    rows, cols, depth = hsi.shape
    data_table = hsi.reshape(rows*cols, depth)
    skewers = np.random.rand(depth, n_skewers)
    votes = np.zeros(rows*cols)
    for j in range(n_skewers):
        res = np.abs(np.sum(skewers[:, j] * data_table, axis=1))
        indx = np.argmax(res)
        votes[indx] += 1
    
    votes = sorted(enumerate(votes), key=lambda el: el[1])
    q_votes = np.array(votes[-q:], dtype=int)
    thr_votes = q_votes[q_votes[:, 1] >= thr]
    indx = thr_votes[:, 0]
    answer = data_table[indx]
    return answer