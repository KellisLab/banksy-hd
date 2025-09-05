
def FindMeanVariance(adata, W_table, weight="n_counts_adjusted", batch_size:int=50000, inplace:bool=True,
                     use_lag:bool=True, use_laplacian_of_gaussian:bool=True,
                     use_residuals:bool=True):
    import numpy as np
    import pandas as pd
    from tqdm.auto import tqdm
    from .gather import gather_X
    mean_list = pd.DataFrame(index=adata.var_names)
    squared_mean_list = pd.DataFrame(index=adata.var_names)
    if weight in adata.obs.columns:
        print("Using weight \"%s\" for PCA" % weight)
        w = adata.obs[weight].values
    else:
        print("Using uniform weight for PCA")
        w = np.ones(adata.shape[0])
    w = w / w.sum()
    print("Finding mean and variance")
    for left in tqdm(np.arange(0, adata.shape[0], batch_size)):
        right = min(left + batch_size, adata.shape[0])
        P_list, P_names = gather_X(adata.X,
                                   W_table=W_table,
                                   indices=np.arange(left, right),
                                   use_lag=use_lag, use_laplacian_of_gaussian=use_laplacian_of_gaussian,
                                   use_residuals=use_residuals, combine=False)
        for PN, PX in zip(P_names, P_list):
            if PN not in mean_list.columns:
                mean_list[PN] = 0.
                squared_mean_list[PN] = 0.
            mean_list[PN] += np.ravel(PX.T.dot(w[left:right]))
            #mean_list[PN] += np.ravel(PX.sum(0) / adata.shape[0])
            squared_mean_list[PN] += np.ravel(PX.multiply(PX).T.dot(w[left:right]))
            #squared_mean_list[PN] += np.ravel(PX.multiply(PX).sum(0) / adata.shape[0])
        del P_list, P_names
    var = np.clip(squared_mean_list.values - mean_list.values * mean_list.values, 0, np.inf)
    del squared_mean_list
    std = pd.DataFrame(np.sqrt(var), index=mean_list.index, columns=[f"{x}_std" for x in mean_list.columns])
    mean_list.columns = [f"{x}_mean" for x in mean_list.columns]
    df = pd.concat((mean_list, std), axis=1)
    del mean_list, std
    if inplace:
        for cn in df.columns:
            adata.var[cn] = df[cn]
    else:
        return df
