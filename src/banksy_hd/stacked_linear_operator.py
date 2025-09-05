from scipy.sparse.linalg import LinearOperator
import numpy as np

class StackedLinearOperator(LinearOperator):
    def __init__(self, A: LinearOperator, B: np.ndarray):
        if A.shape[1] != B.shape[1]:
            raise ValueError("Incompatible column dimensions")
        self.A = A
        self.B = B
        m, n = A.shape[0] + B.shape[0], A.shape[1]
        dtype = np.result_type(A.dtype, B.dtype)
        super().__init__(dtype=dtype, shape=(m, n))
    def _matvec(self, x):
        top = self.A._matvec(x)
        bottom = self.B @ x
        return np.concatenate((top, bottom), axis=0)
    def _matmat(self, X):
        top = self.A._matmat(X)
        bottom = self.B @ X
        return np.concatenate((top, bottom), axis=0)
    def _rmatvec(self, x):
        xA = x[:self.A.shape[0]]
        xB = x[self.A.shape[0]:]
        return self.A._rmatvec(xA) + self.B.T @ xB
    def _rmatmat(self, X):
        XA = X[:self.A.shape[0], :]
        XB = X[self.A.shape[0]:, :]
        return self.A._rmatmat(XA) + self.B.T @ XB
