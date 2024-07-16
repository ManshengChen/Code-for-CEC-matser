import scipy.io as sio
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import pairwise_distances

def load_mat(data_name):
    # 'abalone_data.mat'
    # 'abalone_pool_half_vary_k.mat'
    data = sio.loadmat(data_name)
    k_mat = sio.loadmat(data_name[: -9]+'_pool_half_vary_k.mat')
    X = data['data']
    GT = k_mat['gt']
    members = k_mat['members']

    N = members.shape[0]

    poolSize = members.shape[1]
    one = np.ones((N, poolSize), dtype=np.int32)
    members = members - one


    M = 10 # 10 base clusterings
    cntTimes = 1
    bcIdx = np.zeros((cntTimes,M), dtype=np.int32)
    for i in range(cntTimes):
        tmp = np.random.randint(low=0, high=poolSize,size=poolSize,dtype=np.int32)[:M]
        bcIdx[i,:]=tmp

    baseCls= []
    CA = []
    S_v = []
    for iter in range(cntTimes):
        baseCls.append(members[:,bcIdx[iter]])
        CA.append(compute_CA(baseCls[iter]))
        S_v.append(compute_S_v(baseCls[iter]))
    n_views = M


    lamb_v = []
    for v in range(n_views):
        lamb_v.append(1 / n_views)
    return X, baseCls, CA[0][0], n_views, GT, S_v[0], lamb_v, N



def compute_S_v(BP): 
    n = BP.shape[0]
    m = BP.shape[1]
    S_v =[]
    for i in range(m):
        v = BP[:,i]
        s = np.zeros((n,max(v)+1),dtype=np.float32)
        for j in range(n):
            for k in range(max(v)+1):
                if v[j]==k:
                    s[j,k]= 1
        S_v.append(s.dot(s.T))
    return S_v

def compute_CA(BP):  
    n = BP.shape[0]
    m = BP.shape[1]
    CA = np.zeros((n,n),dtype=np.float32)
    for i in range(m):
        v = BP[:,i]
        s = np.zeros((n,max(v)+1),dtype=np.float32)
        for j in range(n):
            for k in range(max(v)+1):
                if v[j]==k:
                    s[j,k]= 1
        CA += s.dot(s.T)
    return CA


def spectral_knn(X, knn_number, n_clusters):
    similarity_matrix = np.exp(-pairwise_distances(X, metric='euclidean'))

    spectral_clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed')
    labels = spectral_clustering.fit_predict(similarity_matrix)

    knn = np.zeros((X.shape[0], knn_number), dtype=int)

    for i in range(X.shape[0]):
        label_i = labels[i]
        similarities_i = similarity_matrix[i, :]
        same_label_indices = np.where(labels == label_i)[0]
        top_similar_indices = np.argsort(-similarities_i)[1:knn_number+1]
        same_label_top_indices = np.intersect1d(same_label_indices, top_similar_indices)
        knn[i, :len(same_label_top_indices)] = same_label_top_indices

    return knn


if __name__ == "__main__":
    load_mat(data_name='./Datasets/binalpha_data.mat') 