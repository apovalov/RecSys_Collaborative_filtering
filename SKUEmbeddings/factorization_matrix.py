from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix
import numpy as np


def items_embeddings(ui_matrix: csr_matrix, dim: int) -> np.ndarray:
    """Build items embedding using factorization model.
    The order of items should be the same in the output matrix.

    Args:
        ui_matrix (csr_matrix): User-Item matrix of size (N, M)
        dim (int): Dimension of embedding vectors

    Returns:
        np.ndarray: Items embeddings matrix of size (M, dim)
    """
    # Perform SVD decomposition
    _, _, vt = svds(ui_matrix, k=dim)

    # VT contains the item embeddings, transpose it
    item_embeddings = vt.T

    return item_embeddings
