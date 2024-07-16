from tqdm import tqdm
import warnings
from KNN import normal_knn
from sklearn.cluster import KMeans, SpectralClustering
import numpy as np
from Adam import adam
# from LoadData_MCGC import *
from Metrics_O2MAC import metrics
from sklearn.metrics.pairwise import cosine_similarity
import yaml
import scipy.io as sio
import math
import scipy.sparse as sp
# from Matlab_Data_loader import load_data_from_matlab
import matplotlib.pyplot as plt
#from load_data import load_mat
from generate_BPs import load_mat,spectral_knn

def My_function(dataname,k,alpha,max_epoch,iter):


    X, BPs, A, n_views, GT, M, lamb_v, N = load_mat(dataname)

    num_cluster = len(np.unique(GT))
    GT = np.array(GT).flatten()

    V = np.zeros((N, N))
    U = np.zeros((N, N))
    I = np.eye(N)

    #------------------construct knn---------------------------
    knn_number = k
    knn = normal_knn(X, knn_number, num_view=1)

    # ------------------------ begin iteration ------------------------
    print('data = {}, iter={}'.format(dataname[11:] ,iter+1))
    for epoch in tqdm(range(max_epoch),ncols=100):
        F_norm = []
        sum_F_norm = 0
        S = cosine_similarity(V)
        grad = np.zeros((N, N))

        cf_1 = None
        cf_2 = None
    # -------------------- U V sub problem ---------------------------------
        U_grad = np.zeros((N,N))
        V_sub_1 = np.zeros((N,N))
        for v in range(n_views):
            U_grad += lamb_v[v]**2 * (-2 * (M[v].dot(V) - U.dot((V.T).dot(V))))
            V_sub_1 += lamb_v[v]**2 * (-2 * ((M[v].T).dot(U) - V.dot(U.T).dot(U)))
            F_norm.append(np.linalg.norm(M[v]-U.dot(V.T)))
            sum_F_norm += 1./ F_norm[v]
        for i in range(N):
            grad_1 = 0
            grad_2 = 0
            k0 = np.exp(S[i]).sum() - np.exp(S[i][i])
            for j in range(i,N):
                grad_1 = grad_1 + V_sub_1[i][j]
                if i != j:
                    if j in knn[i]:
                        grad_2 = grad_2 + (-1 + knn_number * np.exp(S[i][j]) / k0)
                    else:
                        grad_2 = grad_2 + (knn_number * np.exp(S[i][j]) / k0)
                grad[i][j] = grad_1 + alpha * grad_2
                grad[j][i] = grad[i][j]

    # ----------------update------------------------------------------------
        for v in range(n_views):
            lamb_v[v] = (1./F_norm[v]) / (sum_F_norm)
        V, cf_1 = adam(V, grad, cf_1)
        U, cf_2 = adam(U, U_grad, cf_2)

        C = 0.5 * (np.fabs(V) + np.fabs(V.T))
    return C, GT, num_cluster