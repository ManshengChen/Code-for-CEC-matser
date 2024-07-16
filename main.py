from CEC_model import *



max_iteration = 20 # for average
dataname = './Datasets/binalpha_data.mat'
alpha = 1e-5
max_epoch = 100
k = 10


ACC = []
NMI = []
PUR = []
ARI = []
for i in range(max_iteration):
    C, GT, num_cluster = My_function(dataname=dataname, k=k, alpha=alpha, max_epoch=max_epoch,iter=i)
    u, s, v = sp.linalg.svds(C, k=num_cluster, which='LM')
    kmeans = KMeans(n_clusters=num_cluster, random_state=23).fit(u)
    predict_labels = kmeans.predict(u)
    re_ = metrics.clustering_metrics(GT, predict_labels)
    ac, nm, ari, pur = re_.evaluationClusterModelFromLabel()
    ACC.append(ac)
    NMI.append(nm)
    PUR.append(pur)
    ARI.append(ari)
avg_acc = np.mean(ACC)
avg_nmi = np.mean(NMI)
avg_pur = np.mean(PUR)
avg_ari = np.mean(ARI)
std_acc = np.std(ACC, ddof=1)
std_nmi = np.std(NMI, ddof=1)
std_pur = np.std(PUR, ddof=1)
std_ari = np.std(ARI, ddof=1)

print('data_name = {}, alpha = {}, iteration = {},\navg_acc = {:.4f}, avg_nmi = {:.4f}, avg_pur = {:.4f}, avg_ari = {:.4f},\n'
    'std_acc = {:.4f}, std_nmi = {:.4f}, std_pur = {:.4f}, std_ari = {:.4f}'.format(dataname, alpha, max_iteration,
                                                                                    avg_acc, avg_nmi, avg_pur,
                                                                                    avg_ari,
                                                                                    std_acc, std_nmi, std_pur,
                                                                                    std_ari))