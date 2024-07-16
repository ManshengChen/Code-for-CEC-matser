import sklearn.cluster
from sklearn.neighbors import NearestNeighbors
import numpy as np


def initialization_knn(x, folds, knn_number, num_view):

    N = x[0][0].shape[0]
    Z = np.zeros((N,N))

    nbrs_inx = []
    Ws=[]
    for v in range(num_view):
        Z_v = np.zeros((N, N))
        X1 = x[0][v]

        X1 = X1.T

        ind = np.array(folds[v], dtype=X1.dtype)  

        X2 = X1[:, ind.reshape(-1) == 1]  

        W = np.eye(N) * ind
        Ws.append(W)
        Wv = W[:, ind.reshape(-1) == 1] 

        nbrs = NearestNeighbors(n_neighbors=knn_number+1, algorithm='auto').fit(X2.T) 
        cc = nbrs.kneighbors_graph(X2.T)
        a = nbrs.kneighbors_graph(X2.T).toarray() 
        knn_map = np.matmul(np.matmul(Wv, a), Wv.T)


        nbrs_v = np.zeros((N,knn_number))
        for i in range(N):
            knn_map[i,i]= 0
            kk = np.array(knn_map[i,:]).astype(int)  
            kk = np.nonzero(kk)  
            for j in range(knn_number):
                if kk[0].shape[0] != 0:
                    nbrs_v[i][j] = kk[0][j]  
        nbrs_inx.append(np.array(nbrs_v).astype(int))

        N1 = X2.shape[1]
        nbrs_v = np.zeros((N1, knn_number-1)) 
        dis, idx = nbrs.kneighbors(X2.T)

        tj = -1
        for i in range(N):
            if W[i][i] != 0:
                tj = tj + 1
                for k in range(knn_number):
                    g = int(idx[tj][k])
                    Z_v[i][g] = dis[tj][k]
        Z =Z + Z_v

    Z = my_normalization(Z,0)

    return  nbrs_inx, Ws,Z

def normal_knn(x, knn_number, num_view):
    
    nbrs_inx = []
    N = x.shape[0]

    X_nb = np.array(x)
    nbrs_v = np.zeros((N, knn_number))
    nbrs = NearestNeighbors(n_neighbors=knn_number+1, algorithm='auto').fit(X_nb)
    dis, idx = nbrs.kneighbors(X_nb)
    for i in range(N):
        for j in range(knn_number):
            nbrs_v[i][j] += idx[i][j + 1]
        # svaing for cheap computing

    nbrs_inx=(np.array(nbrs_v).astype(int))
    return nbrs_inx
def my_normalization(data, dim):
    # data.shape=[x,y] dim=0 -> x
    norm = np.sum(data ** 2, axis=dim, keepdims=True)
    norm[norm == 0] = 1.
    return data * np.power(norm, -0.5)  # xi = 0  1/x^-.5 = inf
    # return (data - np.min(data)) / _range


"""
 iv = 1:num_Views
    X1 = X{iv}; %X1∈mv*n
    X1 = NormalizeFea(X1,0);
    ind_0 = find(ind_folds(:,iv) == 0);
    X1(:,ind_0) = 0;    % 缺失视角补0 得到缺失样本后 构建knn图
    Y{iv} = X1;         % 保存缺失数据
    % ---------- 初始KNN图构建 ----------- %
    X1(:,ind_0) = []; %X缺失数据部分为空
    options = [];
    options.NeighborMode = 'KNN';
    options.k = 11;
    options.WeightMode = 'Binary';      % Binary  HeatKernel
    tmpknn = full(constructW(X1',options));  %对完整数据构造KNN图完成初始化,此时Z1大小为完整数据nv*nv
    linshi_W = diag(ind_folds(:,iv)); % ind_folds 对角化
    linshi_W(:,ind_0) = [];
    Knn{iv} = linshi_W * tmpknn * linshi_W'; %得到每个视图的KNN
    W{iv} = eye(num_samples);
    W{iv}(ind_0,ind_0)=0; % W为索引矩阵
"""

