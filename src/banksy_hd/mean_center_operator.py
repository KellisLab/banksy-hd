import numpy as np
from numpy.typing import NDArray
from scipy.sparse import spmatrix, diags
from scipy.sparse.linalg import LinearOperator
from typing import Union

class MeanCenterOperator(LinearOperator):
    def __init__(self, X:Union[spmatrix, LinearOperator], mu:NDArray[np.float64], std:NDArray[np.float64], epsilon:float=1e-300):
        self.X = X
        self.mu = np.ravel(mu).reshape(1, -1)
        self.inv_std = np.ravel(1. / np.clip(std, epsilon, np.inf).astype(np.float64))
        if len(mu) != len(std):
            raise ValueError("Incorrect sizes")
        dtype = X.dtype if np.issubdtype(X.dtype, np.floating) else np.float64
        super().__init__(dtype=dtype, shape=X.shape)
    def _matmat(self, M: NDArray[np.float64]) -> NDArray[np.float64]:
        sv = diags(self.inv_std) @ M
        return self.X.dot(sv) - self.mu @ sv
    def _matvec(self, v: NDArray[np.float64]) -> NDArray[np.float64]:
        sv = diags(self.inv_std) @ v
        return self.X.dot(sv) - self.mu @ sv
    def _rmatmat(self, M: NDArray[np.float64]) -> NDArray[np.float64]:
        XTv = self.X.T.dot(M)
        rhs = self.mu.T @ M.sum(0, keepdims=True)
        return diags(self.inv_std) @ (XTv - rhs) 
    def _rmatvec(self, v: NDArray[np.float64]) -> NDArray[np.float64]:
        XTv = self.X.T.dot(v)
        rhs = self.mu.T @ v.sum(0, keepdims=True)
        return diags(self.inv_std) @ (XTv - rhs)
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  shape={self.shape},\n"
            f"  dtype={self.dtype},\n"
            f"  mean={self.mu}\n"
            f"  std={1./self.inv_std}\n"
            f")"
        )
