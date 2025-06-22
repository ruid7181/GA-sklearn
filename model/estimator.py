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

    def __init__(self, **kwargs):
        # ----------------------------------------------------------------
        # GA hyperparameters. Refer to GeoAggregator class docstring for details.
        self.attn_variant = kwargs.get('attn_variant')
        self.model_variant = kwargs.get('model_variant', None)
        self.d_model = kwargs.get('d_model', 32)
        self.n_attn_layer = kwargs.get('n_attn_layer', 2)
        self.idu_points = kwargs.get('idu_points', 4)
        self.seq_len = kwargs.get('seq_len', 128)
        self.attn_dropout = kwargs.get('attn_dropout', 0.05)
        self.attn_bias_factor = kwargs.get('attn_bias_factor', None)
        self.reg_lin_dims = kwargs.get('reg_lin_dims', None)
        # ----------------------------------------------------------------
        # Training settings.
        self.epochs = kwargs.get('epochs', 20)
        self.lr = kwargs.get('lr', 5e-3)
        self.batch_size = kwargs.get('batch_size', 8)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.verbose = kwargs.get('verbose', True)

        self.model = None
        self.tab_sampler = None
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
            {"Training on device":<30}{str(self.device):>18}
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
                                   dc_lin_dims=self.reg_lin_dims)

        # Fit the model
        _train_ga_regressor(model=self.model,
                            train_loader=train_loader,
                            max_lr=self.lr,
                            epochs=self.epochs,
                            device=self.device,
                            verbose=self.verbose)

    def predict(self, X, l, n_estimate=8, get_std=False, verbose=True):
        """
        :param X:
            co-variates of the tabular dataset.
        :param l:
            2D spatial locations.
        """
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
                                  device=self.device,
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
                                      device=self.device,
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

        X_checked = check_array(
            arr,
            ensure_2d=ensure_2d,
            allow_nd=allow_nd,
            force_all_finite=force_all_finite
        )

        if orig_cols is not None:
            df = pd.DataFrame(X_checked, index=orig_index, columns=orig_cols)
        else:
            default_cols = columns
            df = pd.DataFrame(X_checked, columns=default_cols)
        return df
