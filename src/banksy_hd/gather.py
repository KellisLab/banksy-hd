
from scipy.sparse import spmatrix
from numpy.typing import NDArray
import numpy as np
from typing import Dict, Union

def gather_X(X:spmatrix, W_table:Dict[str, spmatrix],
             indices:NDArray[np.int64],
             mask:Union[None, NDArray[bool], NDArray[np.int64]]=None,
             combine:bool=False,
             use_lag:bool=True, use_laplacian_of_gaussian:bool=True, use_residuals:bool=True):
    import numpy as np
    import scipy.sparse
    Xind = X[indices, :]
    out = []
    out_names = ["Raw"]
    if mask is None:
        out.append(Xind)
    else:
        out.append(Xind[:, mask])
    for WN, W in W_table.items():
        if mask is None:
            WX = W[indices, :] @ X
        else:
            WX = W[indices, :] @ X[:, mask]
        if use_lag:
            out.append(WX)
            out_names.append(f"Lag{WN}")
        if use_laplacian_of_gaussian:
            W2 = W[indices, :] @ W
            if mask is None:
                LoG = WX - W2 @ X
            else:
                LoG = WX - W2 @ X[:, mask]
            out.append(LoG)
            out_names.append(f"LoG{WN}")
        if use_residuals:
            Res = out[0] - WX
            out.append(Res)
            out_names.append(f"Res{WN}")
    if combine:
        return scipy.sparse.hstack(out), out_names
    else:
        return out, out_names
