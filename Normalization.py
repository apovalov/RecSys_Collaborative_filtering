from scipy.sparse import csr_matrix, diags
import numpy as np


class Normalization:

    @staticmethod
    def by_column(matrix: csr_matrix) -> csr_matrix:
        # Calculate the column-wise sum
        col_sum = matrix.sum(axis=0).flatten()
        # Avoid division by zero
        col_sum[col_sum == 0] = 1.0
        # Normalize by column sum
        norm_matrix = csr_matrix(matrix.multiply(1 / col_sum))
        return norm_matrix

    @staticmethod
    def by_row(matrix: csr_matrix) -> csr_matrix:
        """Normalization by row

        Args:
            matrix (csr_matrix): User-Item matrix of size (N, M)

        Returns:
            csr_matrix: Normalized matrix of size (N, M)
        """
        row_sum = matrix.sum(axis=1)
        row_sum[row_sum == 0] = 1  # Avoid division by zero
        norm_matrix = csr_matrix(matrix.multiply(1 / row_sum))
        return norm_matrix

    @staticmethod
    def tf_idf(matrix: csr_matrix) -> csr_matrix:
        """TF-IDF transformation for a term-document matrix

        Args:
            matrix (csr_matrix): Term-document matrix of size (N, M)

        Returns:
            csr_matrix: TF-IDF transformed matrix of size (N, M)
        """

        N, _ = matrix.shape

        # Calculate the inverse document frequency (IDF) for each term
        document_freq = matrix.astype(bool).sum(axis=0).A1
        idf = np.log(N / document_freq)

        term_freq = matrix.sum(axis=1).A1

        # Create a diagonal matrix for IDF
        idf_diag = diags(idf, 0, format='csr')

        # Calculate the TF-IDF matrix
        tf_matrix = matrix.multiply(1 / term_freq[:, np.newaxis])
        tfidf_matrix = tf_matrix.dot(idf_diag)

        return tfidf_matrix

    @staticmethod
    def bm_25(matrix: csr_matrix, k1: float = 2.0, b: float = 0.75) -> csr_matrix:

        N, M = matrix.shape

        # Sum of the all columns for each row [[3] [1] [3] [4]]
        row_sum = matrix.sum(axis=1)

        # Mean 2.75
        avgdl = row_sum.mean()

        # Number of non null rows in each column  [3 3 1]
        document_freq = matrix.astype(bool).sum(axis=0).A1

        # Sun of all columns for each row [3 1 3 4]
        term_freq = matrix.sum(axis=1).A1

        # [0.28768207 0.28768207 1.38629436]
        idf = np.log(N / document_freq)
        rows, cols = matrix.nonzero()

        # Length of the each document, something like this: [[3] [1] [3] [4]]
        dl = row_sum  # [rows]
        delta = k1 * ((1 - b) + b * dl / avgdl)

        idf_diag = diags(idf, 0, format='csr')

        # Calculate the TF matrix
        tf_matrix = matrix.multiply(1 / term_freq[:, np.newaxis])
        tf_matrix.dtype = float

        tfupd = tf_matrix.multiply(1 / delta)

        tfupd = tfupd.power(-1)
        tfupd.data += 1
        tfupd = tfupd.power(-1) * (k1 + 1)

        # Multiply with the IDF matrix
        bm25_matrix = tfupd.dot(idf_diag)

        return bm25_matrix
