import numpy as np 
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt 

def cov_to_corr(cov):
    # Calculate the standard deviations
    std = np.sqrt(np.diag(cov))

    # Calculate the correlation matrix
    corr = cov / np.outer(std, std)

    return corr

def group_corr_inds(corr):

    # print(corr)
    # X = df.corr().values
    X = corr 
    # print(X.shape)
    d = sch.distance.pdist(X)   # vector of ('55' choose 2) pairwise distances
    L = sch.linkage(d, method='complete')
    ind = sch.fcluster(L, 0.5*d.max(), 'distance')
    return ind

def visualize_covmat(cov,plot_names):
    corr = cov_to_corr(cov)
    ind = group_corr_inds(corr)
    
    args = np.argsort(ind)
    sorted_labels = [plot_names[i] for i in args]
    plt.imshow(corr[args, :][:, args], origin="upper")
    plt.xticks(range(len(plot_names)), sorted_labels, rotation=45)
    plt.yticks(range(len(plot_names)), sorted_labels)
    plt.colorbar()
