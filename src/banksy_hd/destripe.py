
def sinkhorn_knopp(X, row_tgt=None, col_tgt=None,
                   n_iter:int=100, tolerance:float=1e-6, eps:float=1e-20):
    import numpy as np
    if row_tgt is not None:
        N = row_tgt
    else:
        N = np.ones(X.shape[0])
    if col_tgt is not None:
        M = col_tgt
    else:
        M = np.ones(X.shape[1])
    u = np.repeat(1, X.shape[0])
    for i in range(n_iter):
        v = M / X.T.dot(u).clip(eps, np.inf)
        u = N / X.dot(v).clip(eps, np.inf)
        if i % 10 == 0 or i + 1 == n_iter:
            row_error = np.abs(M - v * X.T.dot(u)).max()
            col_error = np.abs(N - u * X.dot(v)).max()
            if (row_error < tolerance) and (col_error < tolerance):
                break
    return u, v

def destripe(adata, radius:int=4, col_name='n_counts_sk_adjusted'):
    import numpy as np
    from scipy.ndimage import gaussian_filter
    from scipy.sparse import csr_matrix
    import pandas as pd
    rows = adata.obs["array_row"].values
    cols = adata.obs["array_col"].values
    counts = adata.obs["n_counts"].astype(int)
    n_rows, n_cols = rows.max() + 1, cols.max() + 1
    C_idx = csr_matrix((np.ones_like(counts),
                        (rows, cols)),
                       shape=(n_rows, n_cols),
                       dtype=int).toarray()
    if C_idx.max() > 1:
        print("Warning: %d duplicate rows detected, perhaps this should be done split by sample" % (C > 1).sum())
    del C_idx
    C = csr_matrix((counts, (rows, cols)),
                   shape=(n_rows, n_cols),
                   dtype=int).toarray()
    S = np.expm1(gaussian_filter(np.log1p(C), sigma))
    S[S <= 0.] = np.median(S[S > 0])
    M = (C > 0).astype(bool)
    row_tgt = M.sum(1).astype(float)
    row_tgt[row_tgt <= 0.] = row_tgt[row_tgt > 0].min()
    col_tgt = M.sum(0).astype(float)
    col_tgt[col_tgt <= 0.] = col_tgt[col_tgt > 0].min()
    row_tgt *= col_tgt.sum() / row_tgt.sum()
    alpha, beta = sinkhorn_knopp(C / S,
                                 row_tgt=row_tgt,
                                 col_tgt=col_tgt)
    adj = np.outer(alpha, beta) * C
    adata.obs[col_name] = adj[rows, cols]

    



