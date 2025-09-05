from numpy.typing import NDArray
import numpy as np
def spatial_gaussian_kernel(distances, quantiles:NDArray[np.float64], diag:bool=True):
    import numpy as np
    from scipy.sparse import csr_matrix, diags
    C = distances.copy()
    C.eliminate_zeros()
    Q = np.ones((C.shape[0], len(quantiles)))
    for i in range(C.shape[0]):
        row = C.data[C.indptr[i]:C.indptr[i+1]]
        if len(row) > 0:
            Q[i, :] = np.quantile(row, quantiles)
    tbl = {}
    for i, quant in enumerate(quantiles):
        M = diags(1/Q[:,i]).dot(C).tocsr()
        M.data = np.exp(-1 * M.data**2 / 2)
        if diag:
            M.setdiag(1)
        else:
            M.setdiag(0)
        row_sum = np.ravel(M.sum(1)) + 1e-300
        tbl["Q%.2f" % quant] = diags(1/row_sum) @ M
        del M
    return tbl
