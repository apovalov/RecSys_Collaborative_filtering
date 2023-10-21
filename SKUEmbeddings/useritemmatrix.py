import pandas as pd
from typing import Dict

from scipy.sparse import csr_matrix
# from collections import defaultdict

class UserItemMatrix:
    def __init__(self, sales_data: pd.DataFrame):
        """Class initialization. You can make necessary
        calculations here.

        Args:
            sales_data (pd.DataFrame): Sales dataset.

        Example:
            sales_data (pd.DataFrame):

                user_id  item_id  qty   price
            0        1      118    1   626.66
            1        1      285    1  1016.57
            2        2     1229    3   518.99
            3        4     1688    2   940.84
            4        5     2068    1   571.36
            ...

        """
        self._sales_data = sales_data.copy()

        # # Calculate user and item counts

        unique_users = list(sorted(self._sales_data['user_id'].unique())) #list of unique users
        unique_items = list(sorted(self._sales_data['item_id'].unique()))

        # Calculate user and item counts
        self._user_count = len(unique_users)
        self._item_count = len(unique_items)

        # Create user and item maps
        self._user_map = {user_id: idx for idx, user_id in enumerate(unique_users)}
        self._item_map = {item_id: idx for idx, item_id in enumerate(unique_items)}

        # Create sparse matrix
        rows = [self._user_map[user_id] for user_id in self._sales_data['user_id']]
        cols = [self._item_map[item_id] for item_id in self._sales_data['item_id']]
        self._matrix = csr_matrix((self._sales_data['qty'], (rows, cols)), shape=(self._user_count, self._item_count))

    @property
    def user_count(self) -> int:
        """
        Returns:
            int: the number of users in sales_data.
        """
        return self._user_count

    @property
    def item_count(self) -> int:
        """
        Returns:
            int: the number of items in sales_data.
        """
        return self._item_count

    @property
    def user_map(self) -> Dict[int, int]:
        """Creates a mapping from user_id to matrix rows indexes.

        Example:
            sales_data (pd.DataFrame):

                user_id  item_id  qty   price
            0        1      118    1   626.66
            1        1      285    1  1016.57
            2        2     1229    3   518.99
            3        4     1688    2   940.84
            4        5     2068    1   571.36

            user_map (Dict[int, int]):
                {1: 0, 2: 1, 4: 2, 5: 3}

        Returns:
            Dict[int, int]: User map
        """
        return dict(sorted(self._user_map.items()))

    @property
    def item_map(self) -> Dict[int, int]:
        """Creates a mapping from item_id to matrix rows indexes.

        Example:
            sales_data (pd.DataFrame):

                user_id  item_id  qty   price
            0        1      118    1   626.66
            1        1      285    1  1016.57
            2        2     1229    3   518.99
            3        4     1688    2   940.84
            4        5     2068    1   571.36

            item_map (Dict[int, int]):
                {118: 0, 285: 1, 1229: 2, 1688: 3, 2068: 4}

        Returns:
            Dict[int, int]: Item map
        """
        return dict(sorted(self._item_map.items()))

    @property
    def csr_matrix(self) -> csr_matrix:
        """User items matrix in form of CSR matrix.

        User row_ind, col_ind as
        rows and cols indecies(mapped from user/item map).

        Returns:
            csr_matrix: CSR matrix
        """
        return self._matrix


# sales = pd.read_excel('3_8_SKU_EMB_2023_08_15.xlsx')
# sales_df = pd.DataFrame(sales)
# # # print(sales_df.head())

# # unique_users = sales_df['user_id'].unique()
# # item_map = {item_id: idx for idx, item_id in enumerate(unique_users)}

# # print(dict(sorted(item_map.items())))


# itemMatrix = UserItemMatrix(sales_df)
# print(itemMatrix.item_map.keys())