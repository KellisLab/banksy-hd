from typing import List, Union
from numpy.typing import NDArray
from scipy.sparse import diags

def FitPCA(adata, quantiles:Union[NDArray[float], List[float]],
           n_components:int=50, weight:str="_NullWeight",
           batch_size:int=50000, lamb:float=0.2, mu:float=1.5, rho:float=5,
           mask_var:str="",
           use_lag:bool=True,
           use_laplacian_of_gaussian:bool=True,
           use_residuals:bool=False,
           keep_pc_loadings:bool=True,
           kernel_use_diag:bool=False):
    """
    todo needs docstrings.
    Lambda: ratio between X and spatial (as in Banksy)
    Mu: decimal odds between Lag and LoG
    Rho: decimal odds between residuals and Lag + LoG.
    Usage: FitPCA(adata, [0.5])
    Make sure adata is normalized beforehand (log, size factor adjusted, etc)
    and spatial_neighbors(adata, library_id="slice") is called already
    """
    import numpy as np
    from scipy.sparse import csr_matrix
    import pandas as pd
    from tqdm.auto import tqdm
    from .kernel import spatial_gaussian_kernel
    from .ipcas import IPCAS
    from .gather import gather_X
    from .moments import FindMeanVariance
    ### First, fix rho, mu in case of missing matrices
    if not use_residuals:
        rho = 1.
    elif not laplacian_of_gaussian and not use_lag:
        rho = np.inf
    if not use_laplacian_of_gaussian:
        mu = 1.
    elif not use_lag:
        mu = np.inf
    if weight not in adata.obs.columns:
        if weight is not None:
            print("Warning: Not using weight \"%s\" since not in columns" % weight)
        weight = None
    W = spatial_gaussian_kernel(adata.obsp["spatial_distances"],
                                quantiles=quantiles, diag=kernel_use_diag)
    msf = FindMeanVariance(adata, W, weight=weight, batch_size=batch_size, inplace=False,
                           use_lag=use_lag, use_laplacian_of_gaussian=use_laplacian_of_gaussian,
                           use_residuals=use_residuals)
    P_list = []
    for cn in msf.columns:
        if cn.endswith("_mean") and cn.replace("_mean", "_std") in msf.columns:
            P_list.append(cn.replace("_mean", ""))
        adata.var[cn] = msf[cn]
        adata.var[cn.replace("_mean", "_std")] = msf[cn.replace("_mean", "_std")]
    mask = msf["Raw_std"].values > 0 ### raw is always 1st
    if mask_var in adata.var.columns:
        mask = adata.var[mask_var].values & mask
    pf = pd.DataFrame(index=P_list)
    pf["Lag"] = pf.index.str.startswith("Lag")
    pf["LoG"] = pf.index.str.startswith("LoG")
    pf["Res"] = pf.index.str.startswith("Res")
    ## get_dummies(pf.index.str.replace(("LoG", "Res", "Lag"), ""))
    pf["Weight"] = np.sqrt(1-lamb)
    pf.loc[pf["Lag"], "Weight"] = np.sqrt(lamb * (1 / mu) * (1 / rho) / len(W))
    pf.loc[pf["LoG"], "Weight"] = np.sqrt(lamb * (1 - 1/mu) * (1 / rho) / len(W))
    pf.loc[pf["Res"], "Weight"] = np.sqrt(lamb * (1 - 1/rho) / len(W))
    ### debug: sum the squared pf['Weight'] and check if equal to 1
    mean, std = [], []
    for cn in pf.index:
        mean += list(msf[f"{cn}_mean"].values[mask])
        std += list(msf[f"{cn}_std"].values[mask] / pf.loc[cn, "Weight"])
    pca = IPCAS(n_components, np.asarray(mean), np.asarray(std))
    rand_indices = np.random.permutation(adata.shape[0])
    w = np.ones(adata.shape[0])
    if weight is not None:
        w = adata.obs[weight].values
    w = w / w.mean()
    n_eff = w.sum()**2/np.sum(w*w)
    gamma = pf.shape[0] * mask.sum() / n_eff
    print("gamma:", gamma)
    lamb_max = pf.shape[0] * pf["Weight"].max()**2 * (1 + np.sqrt(gamma))**2
    print("lambda max:", lamb_max)
    print("Effective N:", n_eff, "vs original N:", adata.shape[0])
    print("PF:")
    print(pf)
    print("Fitting PCA")
    for left in tqdm(np.arange(0, adata.shape[0], batch_size)):
        right = min(left + batch_size, adata.shape[0])
        ind = rand_indices[left:right] ### Shuffle indices for PCA
        X, P_names = gather_X(adata.X, W, indices=ind, mask=mask, combine=True,
                              use_lag=use_lag, use_residuals=use_residuals,
                              use_laplacian_of_gaussian=use_laplacian_of_gaussian)
        ### TODO assert P_names correspond to pf.index
        ### Lanczos weights must be \sqrt of original weights
        pca.partial_fit(X, w=np.sqrt(w[ind]))
        del X
    adata.uns["pca"] = {"variance": pca.explained_variance_ * pca.n_samples_seen_ / n_eff,
                        "variance_ratio": pca.explained_variance_ratio_ * pca.n_samples_seen_ / n_eff,
                        "mp_bulk_max": lamb_max}
    ratio = np.sqrt(((adata.uns["pca"]["variance"]**2 - gamma - 1)**2 - 4 * gamma) * (adata.uns["pca"]["variance"] >= lamb_max)) * adata.uns["pca"]["variance"]**-2
    adata.uns["pca"]["mp_ratio"] = ratio.copy()
    n_mp = int(np.sum(adata.uns["pca"]["variance"] >= lamb_max))
    ratio = csr_matrix(ratio)[:, :n_mp]
    print("# of PCs passing MP threshold:", n_mp)
    print("PCs:", adata.uns["pca"]["variance"])
    print("MP threshold:", lamb_max)
    nmask = int(mask.sum())
    adata.varm["PCs"] = np.zeros((adata.shape[1], n_components), dtype=np.float32)
    adata.varm["PCs"][mask, :] = pca.components_[:, :nmask].T
    if keep_pc_loadings:
        for pf_idx in range(1, pf.shape[0]):
            left = pf_idx * nmask
            right = left + nmask
            pname = pf.index.values[pf_idx]
            adata.varm[f"PCs_{pname}"] = np.zeros((adata.shape[1], n_components), dtype=np.float32)
            adata.varm[f"PCs_{pname}"][mask, :] = pca.components_[:, left:right].T
    print("Expanding PCA")
    adata.obsm["X_pca"] = np.zeros((adata.shape[0], n_components), dtype=np.float32)
    sv = diags(1. / (1e-100 + adata.var["Raw_std"].values)) @ adata.varm["PCs"] 
    adata.obsm["R_pca"] = (adata.X.dot(sv) - adata.var["Raw_mean"].values[None, :] @ sv).astype(np.float32)
    for left in tqdm(np.arange(0, adata.shape[0], batch_size)):
        right = min(left + batch_size, adata.shape[0])
        X, P_names = gather_X(adata.X, W, indices=np.arange(left, right), mask=mask, combine=True,
                              use_lag=use_lag, use_residuals=use_residuals,
                              use_laplacian_of_gaussian=use_laplacian_of_gaussian)
        XP = pca.transform(X) #@ diags(ratio)
        adata.obsm["X_pca"][left:right, :] = XP#[:, :n_mp]
        del X, XP
        

