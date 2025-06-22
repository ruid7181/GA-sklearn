import logging
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.neighbors import NearestNeighbors


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('[%(levelname)s] %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

sampler_mode = Literal['train', 'val', 'explain']


class TabDataSampler(Dataset):
    def __init__(self,
                 sample_radius=None,
                 radius_estimate_ratio=0.5,
                 seq_len=None,
                 x_cols=None,
                 spa_cols=None,
                 y_cols=None,
                 device=None):
        """
        :param sample_radius:

        """
        self.s_radius = sample_radius
        self.radius_estimate_ratio = radius_estimate_ratio
        self.seq_len = seq_len
        self.x_cols = x_cols
        self.spa_cols = spa_cols
        self.y_cols = y_cols
        self.feat_cols = list(x_cols) + list(spa_cols) + list(y_cols)
        self.device = device

        self.sampler_mode: sampler_mode = 'train'
        self.query_tree = None
        self.context_pool, self.query_pool = None, None
        self._context_tensor_pool = None
        self._query_tensor_pool = None
        self._neighbors_cache, self._dists_cache = None, None

        self.x_cols_id = np.arange(
            len(self.x_cols)
        )
        self.spa_cols_id = np.arange(
            len(self.x_cols), len(self.x_cols) + len(self.spa_cols)
        )
        self.y_cols_id = np.array([-1])

        self.explain_call_count = -2
        self.n_background = None

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
        assert (self._neighbors_cache is not None
                and self._dists_cache is not None), ('Need to provide neighborhood relationship, '
                                                     'by calling `pre_compute_neighbors()`.')

        if self.sampler_mode == 'explain':
            q_item_in_cache = self.explain_call_count // self.n_background
        else:
            q_item_in_cache = item

        # Query, retrieve cached context points
        indices = self._neighbors_cache[q_item_in_cache]
        dists = self._dists_cache[q_item_in_cache]
        if self.sampler_mode == 'train':
            # Remove target point itself from retrieved points.
            dists = dists[indices != q_item_in_cache]
            indices = indices[indices != q_item_in_cache]

        n_neighbor = indices.shape[0]

        if n_neighbor <= self.seq_len:
            # h^{in} <= l_{max}, zero padding
            sample = torch.zeros(
                size=(self.seq_len, len(self.feat_cols)),
                dtype=torch.float,
                device=self.device
            ) + torch.nan
            sample[:n_neighbor] = self._context_tensor_pool[indices]
            sample[-1] = self._query_tensor_pool[item: item + 1]

            sample_dist = torch.zeros(
                size=(self.seq_len,),
                dtype=torch.float,
                device=self.device
            ) + torch.nan
            sample_dist[:n_neighbor] = dists
            sample_dist = torch.where(sample_dist == 0, 1e-4, sample_dist)
            sample_dist[-1] = 0.
        else:
            # h^{in} > l_{max}, random clipping
            perm = torch.randperm(
                n_neighbor,
                dtype=torch.long,
                device=self.device
            )[:self.seq_len]
            indices = indices[perm]
            sample = self._context_tensor_pool[indices]
            sample[-1] = self._query_tensor_pool[item: item + 1]

            sample_dist = dists[perm]
            sample_dist = torch.where(sample_dist == 0, 1e-4, sample_dist)
            sample_dist[-1] = 0.

        return sample, sample_dist

    def __len__(self):
        return self.query_pool.shape[0]

    def set_context_pool(self, context_pool: pd.DataFrame = None):
        """
        - Build query tree.
        - Prepare pytorch Tensor dataset from pandas DataFrame.
        - Radius estimation for ContextQuery.
        """
        self.context_pool = context_pool[self.feat_cols].to_numpy()
        self._context_tensor_pool = torch.from_numpy(
            self.context_pool
        ).float().to(self.device)

        self.query_tree = NearestNeighbors(
            algorithm='kd_tree',
            n_jobs=-1
        )
        self.query_tree.fit(self.context_pool[:, self.spa_cols_id])

        if self.s_radius is None:
            self.s_radius = self.__estimate_radius(
                all_points=self.context_pool[:, self.spa_cols_id],
                seq_len=self.seq_len,
                sample_ratio=self.radius_estimate_ratio
            )

    def set_query_pool(self, query_pool: pd.DataFrame = None, pre_compute_neighbors=True):
        """
        - Prepare query data.
        """
        self.query_pool = query_pool[self.feat_cols].to_numpy()
        self._query_tensor_pool = torch.from_numpy(
            self.query_pool
        ).float().to(self.device)

        if self.sampler_mode == 'explain':
            self.explain_call_count += 1

        if pre_compute_neighbors:
            self.__pre_compute_neighbors()

    def __pre_compute_neighbors(self):
        self._dists_cache, self._neighbors_cache = self.query_tree.radius_neighbors(
            X=self.query_pool[:, self.spa_cols_id],
            radius=self.s_radius,
            return_distance=True
        )
        for i in range(len(self._dists_cache)):
            self._dists_cache[i] = torch.from_numpy(self._dists_cache[i]).float().to(self.device)
            self._neighbors_cache[i] = torch.from_numpy(self._neighbors_cache[i]).long().to(self.device)

    def train_mode(self):
        self.sampler_mode = 'train'

    def val_mode(self):
        self.sampler_mode = 'val'

    def explain_mode(self, n_background=30):
        self.sampler_mode = 'explain'
        self.n_background = n_background

    def __count_avg_neighbors(self, all_points, radius, sample_size):
        N = all_points.shape[0]
        if sample_size < N:
            idx = np.random.choice(N, sample_size, replace=False)
            all_points = all_points[idx]

        neighbors_idx = self.query_tree.radius_neighbors(
            all_points,
            radius,
            return_distance=False
        )
        counts = [len(nbrs) - 1 for nbrs in neighbors_idx]

        return np.mean(counts)

    def __estimate_radius(self,
                          all_points: np.array,
                          seq_len=81,
                          sample_ratio=0.3,
                          max_iter=30,
                          tolerance=0.02):
        """
        Find a radius such that the average number of neighbors is approximately 'seq_len'.
        A binary search approach.

        :param all_points:
            a np.ndarray containing only 'spa' features (spatial coordinates).
        :param seq_len:
            expected sequence length.
        :param sample_ratio:
            sample part of the data set for estimation.
        :param max_iter:
            maximum number of iterations.
        :param tolerance:
            stopping criteria.
        """
        seq_len = int(seq_len * 1.25)
        sample_size = int(sample_ratio * len(all_points))

        left, right = 1e-6, 1
        for i in range(max_iter):
            mid = (left + right) / 2.
            avg_seq_len = self.__count_avg_neighbors(all_points=all_points,
                                                     radius=mid,
                                                     sample_size=sample_size)

            if abs(avg_seq_len - seq_len) <= tolerance:
                logger.info(f'Radius estimation ends after {i} iterations. '
                            f'Estimated radius: {mid:.5f} (seq_len extended by 1.25).')
                return mid

            if avg_seq_len > seq_len:
                right = mid
            else:
                left = mid

        logger.info(f'Radius estimation ends after {max_iter} iterations. '
                    f'Estimated radius: {((left + right) / 2.):.5f} (seq_len extended by 1.25).')
        return (left + right) / 2.
