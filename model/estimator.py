import inspect

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array
from torch.utils.data import DataLoader

from model.aggregator import GeoAggregator
from model.aggregator_ds import TabDataSampler
from model.aggregator_utils import _train_ga_regressor, _test_ga_regressor


class GARegressor(BaseEstimator, RegressorMixin):
    """
    A sklearn-style wrapper of GeoAggregator for spatial regression.
    """

    def __init__(
            self,
            x_cols=None,
            spa_cols=None,
            y_cols=None,
            attn_variant='MCPA',
            model_variant=None,
            d_model=32,
            n_attn_layer=2,
            idu_points=4,
            seq_len=128,
            attn_dropout=0.05,
            attn_bias_factor=None,
            reg_lin_dims=None,
            epochs=20,
            lr=5e-3,
            batch_size=8,
            device='auto',
            random_state=None,
            verbose=True):
        # ----------------------------------------------------------------
        # GA hyperparameters. Refer to GeoAggregator class docstring for details.
        self.x_cols = x_cols
        self.spa_cols = spa_cols
        self.y_cols = y_cols
        self.attn_variant = attn_variant
        self.model_variant = model_variant
        self.d_model = d_model
        self.n_attn_layer = n_attn_layer
        self.idu_points = idu_points
        self.seq_len = seq_len
        self.attn_dropout = attn_dropout
        self.attn_bias_factor = attn_bias_factor
        self.reg_lin_dims = reg_lin_dims
        # ----------------------------------------------------------------
        # Training settings.
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

        self.model = None
        self.tab_sampler = None
        self.device_ = self.__resolve_device(device)
        # ----------------------------------------------------------------
        # Model Summary
        if self.model_variant is not None:
            print(f'Using the model template: GA-{self.model_variant}.')

        if self.verbose:
            print(f"""
            {f" GeoAggregator Model Summary ":_^50}
            {"attention mechanism type":<30}{self.attn_variant:>18}
            {"d_model":<30}{self.d_model:>18}
            {"# attention layer":<30}{self.n_attn_layer:>18}
            {"# inducing point":<30}{self.idu_points:>18}
            {"# sequence length":<30}{self.seq_len:>18}
            {"regressor neurons":<30}{str(self.reg_lin_dims):>18}
            
            {f" training details ":_^50}
            {"Training on device":<30}{str(self.device_):>18}
            {"attention dropout rate":<30}{self.attn_dropout:>18}
            {"maximum learning rate":<30}{self.lr:>18}
            {"batch_size":<30}{self.batch_size:>18}
            {"# epoch":<30}{self.epochs:>18}
            """)

    def fit(self,
            X: pd.DataFrame,
            l: pd.DataFrame,
            y: pd.DataFrame):
        """
        Sklearn-style interface for training the GeoAggregator Regressor model
        on a geospatial tabular dataset.

        :param X:
            co-variates of the tabular dataset.
        :param l:
            2D spatial locations.
        :param y:
            the target variable.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
            torch.manual_seed(self.random_state)

        # Using Pytorch-style Dataset & DataLoader
        X = self.__check_array_df(arr=X)
        l = self.__check_array_df(arr=l)
        y = self.__check_array_df(arr=y, ensure_2d=False)
        tab_df = pd.merge(X, l, how="inner", left_index=True, right_index=True)
        tab_df = pd.merge(tab_df, y, how="inner", left_index=True, right_index=True)

        self.tab_sampler = TabDataSampler(x_cols=X.columns,
                                          y_cols=y.columns,
                                          spa_cols=l.columns,
                                          seq_len=self.seq_len)
        self.tab_sampler.train_mode()
        self.tab_sampler.set_context_pool(context_pool=tab_df)
        self.tab_sampler.set_query_pool(query_pool=tab_df)

        x_dims = tuple(range(len(X.columns)))
        spa_dims = (-3, -2)
        y_dims = (-1,)

        train_loader = DataLoader(dataset=self.tab_sampler,
                                  batch_size=self.batch_size,
                                  shuffle=True)

        # Pytorch-style GA model initialization
        self.model = GeoAggregator(x_dims=x_dims,
                                   spa_dims=spa_dims,
                                   y_dims=y_dims,
                                   attn_variant=self.attn_variant,
                                   model_variant=self.model_variant,
                                   d_model=self.d_model,
                                   n_attn_layer=self.n_attn_layer,
                                   idu_points=self.idu_points,
                                   attn_dropout=self.attn_dropout,
                                   attn_bias_factor=self.attn_bias_factor,
                                   dc_lin_dims=self.reg_lin_dims).to(self.device_)

        # Fit the model
        _train_ga_regressor(model=self.model,
                            train_loader=train_loader,
                            max_lr=self.lr,
                            epochs=self.epochs,
                            device=self.device_,
                            verbose=self.verbose)
        return self

    def predict(self, X, l, n_estimate=8, get_std=False, verbose=True):
        """
        :param X:
            co-variates of the tabular dataset.
        :param l:
            2D spatial locations.
        """
        if self.tab_sampler is None or self.model is None:
            raise ValueError("GARegressor must be fitted before calling predict().")

        # Using Pytorch-style Dataset & DataLoader
        X = self.__check_array_df(arr=X, columns=self.tab_sampler.x_cols)
        l = self.__check_array_df(arr=l, columns=self.tab_sampler.spa_cols)
        tab_df = pd.merge(X, l, how="inner", left_index=True, right_index=True)
        tab_df[self.tab_sampler.y_cols] = 0.

        assert X.columns.equals(self.tab_sampler.x_cols)
        assert self.model is not None

        self.tab_sampler.val_mode()
        self.tab_sampler.set_query_pool(query_pool=tab_df)

        data_loader = DataLoader(dataset=self.tab_sampler,
                                 batch_size=1,
                                 shuffle=False)

        # Predict
        return _test_ga_regressor(model=self.model,
                                  test_loader=data_loader,
                                  device=self.device_,
                                  n_estimate=n_estimate,
                                  get_std=get_std,
                                  verbose=verbose)

    def get_shap_predictor(self, X, l, n_background=30):
        """
        :param X:
            co-variates of the tabular dataset TO BE EXPLAINED.
        :param l:
            coordinates of the tabular dataset TO BE EXPLAINED.
        :param n_background:
            number of background points in the explanation.
        """
        # Using Pytorch-style Dataset & DataLoader
        X = self.__check_array_df(arr=X, columns=self.tab_sampler.x_cols)
        l = self.__check_array_df(arr=l, columns=self.tab_sampler.spa_cols)
        tab_df = pd.merge(X, l, how="inner", left_index=True, right_index=True)
        tab_df[self.tab_sampler.y_cols] = 0.

        self.tab_sampler.explain_mode(n_background=n_background)
        self.tab_sampler.set_query_pool(query_pool=tab_df)

        def shap_predictor(all_feat):
            """
            :param all_feat:
                Both co-variates and coordinates of the tabular dataset TO BE EXPLAINED.
            """
            all_feat = self.__check_array_df(arr=all_feat,
                                             columns=list(self.tab_sampler.x_cols) + list(self.tab_sampler.spa_cols))
            all_feat[self.tab_sampler.y_cols] = 0.
            self.tab_sampler.set_query_pool(query_pool=all_feat,
                                            pre_compute_neighbors=False)

            data_loader = DataLoader(dataset=self.tab_sampler,
                                     batch_size=1,
                                     shuffle=False)

            return _test_ga_regressor(model=self.model,
                                      test_loader=data_loader,
                                      device=self.device_,
                                      n_estimate=1,
                                      get_std=False,
                                      verbose=False)

        return shap_predictor

    def __check_array_df(self,
                         arr,
                         columns=None,
                         ensure_2d=True,
                         allow_nd=False,
                         force_all_finite=True) -> pd.DataFrame:
        orig_cols = None
        orig_index = None
        if isinstance(arr, pd.DataFrame):
            orig_cols = arr.columns
            orig_index = arr.index
            arr = arr.values

        check_kwargs = {
            'ensure_2d': ensure_2d,
            'allow_nd': allow_nd,
        }
        if 'ensure_all_finite' in inspect.signature(check_array).parameters:
            check_kwargs['ensure_all_finite'] = force_all_finite
        else:
            check_kwargs['force_all_finite'] = force_all_finite

        X_checked = check_array(arr, **check_kwargs)

        if orig_cols is not None:
            df = pd.DataFrame(X_checked, index=orig_index, columns=orig_cols)
        else:
            default_cols = columns
            df = pd.DataFrame(X_checked, columns=default_cols)
        return df

    @staticmethod
    def __resolve_device(device):
        if device != 'auto':
            return torch.device(device)

        if torch.cuda.is_available():
            return torch.device('cuda:0')

        mps_backend = getattr(torch.backends, 'mps', None)
        if mps_backend is not None and mps_backend.is_available():
            return torch.device('mps')

        return torch.device('cpu')
