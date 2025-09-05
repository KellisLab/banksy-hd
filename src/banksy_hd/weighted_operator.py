import numpy as np
from numpy.typing import NDArray
from scipy.sparse import spmatrix, diags
from scipy.sparse.linalg import LinearOperator
from typing import Union

class WeightedOperator(LinearOperator):
    def __init__(self, X:Union[spmatrix, LinearOperator], w:NDArray[np.float64]):
        self.X = X
        self.w = w
        self.W = diags(w)
        if len(w) != X.shape[0]:
            raise ValueError("Incorrect sizes")
        dtype = X.dtype if np.issubdtype(X.dtype, np.floating) else np.float64
        super().__init__(dtype=dtype, shape=X.shape)
    def _matmat(self, M: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.W @ (self.X @ M)
    def _matvec(self, v: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.w * (self.X @ v)
    def _rmatmat(self, M: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.X.T.dot(self.W @ M)
    def _rmatvec(self, v: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.X.T.dot(self.w * v)
