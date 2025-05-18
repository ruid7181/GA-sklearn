import logging

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import KDTree
from sklearn.utils import shuffle
from torch.utils.data import Dataset


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('[%(levelname)s] %(message)s')
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)


class TabDataSampler(Dataset):
    def __init__(self,
                 sample_radius=None,
                 radius_estimate_ratio=0.5,
                 seq_len=None,
                 x_cols=None,
                 spa_cols=None,
                 y_cols=None):
        self.s_radius = sample_radius
        self.radius_estimate_ratio = radius_estimate_ratio
        self.seq_len = seq_len
        self.x_cols = x_cols
        self.spa_cols = spa_cols
        self.y_cols = y_cols
        self.feat_cols = None

        self.is_training = True
        self.query_tree = None
        self.context_pool, self.query_pool = None, None
        self.context_pool_data = None
        self._context_pool_tensor = None
        self._query_pool_tensor = None

    def __getitem__(self, item):
        """
        Return:
             1) input sequence;
             2) geo-proxy; and
             3) the ground truth value of the target variable of the target point.
        """
        assert (self.context_pool is not None
                and self.query_pool is not None), ('Need to provide context and query '
                                                   'data first, by calling '
                                                   '`set_context_pool()` and '
                                                   '`set_query_pool()`.')
        
        indices, dists = self.query_tree.query_radius(
            X=self.query_pool[self.spa_cols][item: item + 1].to_numpy(),
            r=self.s_radius,
            return_distance=True
        )
        indices = indices[0]
        dists = dists[0]
        
        n_neighbor = indices.shape[0]

        if self.is_training:
            dists = dists[indices != item]
            indices = indices[indices != item]
            n_neighbor -= 1

        query_point = self._query_pool_tensor[item:item+1]

        if n_neighbor <= self.seq_len:
            # h^{in} <= l_{max}, zero padding
            sample = torch.zeros(size=(self.seq_len, len(self.feat_cols)), dtype=torch.float) + torch.nan
            sample[:n_neighbor] = self._context_pool_tensor[indices]
            sample[-1] = query_point

            sample_dist = torch.zeros(size=(self.seq_len,), dtype=torch.float) + torch.nan
            sample_dist[:n_neighbor] = torch.FloatTensor(dists)
            sample_dist = torch.where(sample_dist == 0, 1e-3, sample_dist)
            sample_dist[-1] = 0.
        else:
            # h^{in} > l_{max}, random clipping
            random_this_ = np.random.randint(100)
            indices = shuffle(indices, random_state=item + random_this_)[:self.seq_len]
            sample = self._context_pool_tensor[indices]
            sample[-1] = query_point

            sample_dist = shuffle(dists, random_state=item + random_this_)[:self.seq_len]
            sample_dist = torch.FloatTensor(sample_dist)
            sample_dist = torch.where(sample_dist == 0, 1e-3, sample_dist)
            sample_dist[-1] = 0.

        return sample, sample_dist,

    def __len__(self):
        return len(self.query_pool)

    def train(self):
        self.is_training = True

    def val(self):
        self.is_training = False

    def set_context_pool(self, context_pool: pd.DataFrame = None):
        """
        - Build query tree.
        - Prepare pytorch Tensor dataset from pandas DataFrame.
        - Radius estimation for ContextQuery.
        """
        self.context_pool = context_pool
        # Pre-convert data to tensor
        self._context_pool_tensor = torch.from_numpy(
            self.context_pool[self.feat_cols].to_numpy()
        ).float()
        
        self.query_tree = KDTree(self.context_pool[self.spa_cols])

        self.context_pool_data = torch.FloatTensor(
            self.context_pool[self.feat_cols].to_numpy()
        )

        if self.s_radius is None:
            self.s_radius = self.__estimate_radius(
                all_points=self.context_pool[self.spa_cols],
                seq_len=self.seq_len,
                sample_ratio=self.radius_estimate_ratio
            )

    def set_query_pool(self, query_pool: pd.DataFrame = None):
        self.query_pool = query_pool
        # Pre-convert query pool data to tensor for faster access
        self._query_pool_tensor = torch.from_numpy(
            self.query_pool[self.feat_cols].to_numpy()
        ).float()

    def __count_avg_neighbors(self, all_points, radius, sample_size):
        tree = self.query_tree
        N = all_points.shape[0]
        if sample_size < N:
            idx = np.random.choice(N, sample_size, replace=False)
            all_points = all_points[idx]

        neighbors_idx = tree.query_radius(all_points, radius)
        counts = [len(nbrs) - 1 for nbrs in neighbors_idx]

        return np.mean(counts)

    def __estimate_radius(self,
                          all_points: pd.DataFrame,
                          seq_len=81,
                          sample_ratio=0.3,
                          max_iter=30,
                          tolerance=0.02):
        """
        Find a radius such that the average number of neighbors is approximately 'seq_len'.
        A binary search approach.

        :param all_points:
            a df containing only 'spa' columns (spatial coordinates).
        :param seq_len:
            expected sequence length.
        :param sample_ratio:
            sample part of the data set for estimation.
        :param max_iter:
            maximum number of iterations.
        :param tolerance:
            stopping criteria.
        """
        seq_len = int(seq_len * 1.2)
        sample_size = int(sample_ratio * len(all_points))
        all_points = all_points.values

        left, right = 1e-6, 1
        for i in range(max_iter):
            mid = (left + right) / 2.
            avg_seq_len = self.__count_avg_neighbors(all_points=all_points,
                                                     radius=mid,
                                                     sample_size=sample_size)

            if abs(avg_seq_len - seq_len) <= tolerance:
                logger.info(f'Radius estimation ends after {i} iterations. '
                            f'Estimated radius: {mid:.5f} (seq_len extended by 1.2).')
                return mid

            if avg_seq_len > seq_len:
                right = mid
            else:
                left = mid

        logger.info(f'Radius estimation ends after {max_iter} iterations. '
                    f'Estimated radius: {((left + right) / 2.):.5f} (seq_len extended by 1.2).')
        return (left + right) / 2.
