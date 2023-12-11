

class LinearRegression(MultiOutputMixin, RegressorMixin, LinearModel):
    """
    Ordinary least squares Linear Regression.

    LinearRegression fits a linear model with coefficients w = (w1, ..., wp)
    to minimize the residual sum of squares between the observed targets in
    the dataset, and the targets predicted by the linear approximation.

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set
        to False, no intercept will be used in calculations
        (i.e. data is expected to be centered).

    normalize : bool, default=False
        This parameter is ignored when ``fit_intercept`` is set to False.
        If True, the regressors X will be normalized before regression by
        subtracting the mean and dividing by the l2-norm.
        If you wish to standardize, please use
        :class:`~sklearn.preprocessing.StandardScaler` before calling ``fit``
        on an estimator with ``normalize=False``.

        .. deprecated:: 1.0
           `normalize` was deprecated in version 1.0 and will be
           removed in 1.2.

    copy_X : bool, default=True
        If True, X will be copied; else, it may be overwritten.

    n_jobs : int, default=None
        The number of jobs to use for the computation. This will only provide
        speedup in case of sufficiently large problems, that is if firstly
        `n_targets > 1` and secondly `X` is sparse or if `positive` is set
        to `True`. ``None`` means 1 unless in a
        :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors. See :term:`Glossary <n_jobs>` for more details.

    positive : bool, default=False
        When set to ``True``, forces the coefficients to be positive. This
        option is only supported for dense arrays.

        .. versionadded:: 0.24

    Attributes
    ----------
    coef_ : array of shape (n_features, ) or (n_targets, n_features)
        Estimated coefficients for the linear regression problem.
        If multiple targets are passed during the fit (y 2D), this
        is a 2D array of shape (n_targets, n_features), while if only
        one target is passed, this is a 1D array of length n_features.

    rank_ : int
        Rank of matrix `X`. Only available when `X` is dense.

    singular_ : array of shape (min(X, y),)
        Singular values of `X`. Only available when `X` is dense.

    intercept_ : float or array of shape (n_targets,)
        Independent term in the linear model. Set to 0.0 if
        `fit_intercept = False`.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    Ridge : Ridge regression addresses some of the
        problems of Ordinary Least Squares by imposing a penalty on the
        size of the coefficients with l2 regularization.
    Lasso : The Lasso is a linear model that estimates
        sparse coefficients with l1 regularization.
    ElasticNet : Elastic-Net is a linear regression
        model trained with both l1 and l2 -norm regularization of the
        coefficients.

    Notes
    -----
    From the implementation point of view, this is just plain Ordinary
    Least Squares (scipy.linalg.lstsq) or Non Negative Least Squares
    (scipy.optimize.nnls) wrapped as a predictor object.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LinearRegression
    >>> X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    >>> # y = 1 * x_0 + 2 * x_1 + 3
    >>> y = np.dot(X, np.array([1, 2])) + 3
    >>> reg = LinearRegression().fit(X, y)
    >>> reg.score(X, y)
    1.0
    >>> reg.coef_
    array([1., 2.])
    >>> reg.intercept_
    3.0...
    >>> reg.predict(np.array([[3, 5]]))
    array([16.])
    """

    def __init__(
        self,
        *,
        fit_intercept=True,
        normalize="deprecated",
        copy_X=True,
        n_jobs=None,
        positive=False,
    ):
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.n_jobs = n_jobs
        self.positive = positive

    def fit(self, X, y, sample_weight=None):
        """
        Fit linear model.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values. Will be cast to X's dtype if necessary.

        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample.

            .. versionadded:: 0.17
               parameter *sample_weight* support to LinearRegression.

        Returns
        -------
        self : object
            Fitted Estimator.
        """

        _normalize = _deprecate_normalize(
            self.normalize, default=False, estimator_name=self.__class__.__name__
        )

        n_jobs_ = self.n_jobs

        accept_sparse = False if self.positive else ["csr", "csc", "coo"]

        X, y = self._validate_data(
            X, y, accept_sparse=accept_sparse, y_numeric=True, multi_output=True
        )

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)

        X, y, X_offset, y_offset, X_scale = self._preprocess_data(
            X,
            y,
            fit_intercept=self.fit_intercept,
            normalize=_normalize,
            copy=self.copy_X,
            sample_weight=sample_weight,
            return_mean=True,
        )

        if sample_weight is not None:
            # Sample weight can be implemented via a simple rescaling.
            X, y = _rescale_data(X, y, sample_weight)

        if self.positive:
            if y.ndim < 2:
                self.coef_, self._residues = optimize.nnls(X, y)
            else:
                # scipy.optimize.nnls cannot handle y with shape (M, K)
                outs = Parallel(n_jobs=n_jobs_)(
                    delayed(optimize.nnls)(X, y[:, j]) for j in range(y.shape[1])
                )
                self.coef_, self._residues = map(np.vstack, zip(*outs))
        elif sp.issparse(X):
            X_offset_scale = X_offset / X_scale

            def matvec(b):
                return X.dot(b) - b.dot(X_offset_scale)

            def rmatvec(b):
                return X.T.dot(b) - X_offset_scale * np.sum(b)

            X_centered = sparse.linalg.LinearOperator(
                shape=X.shape, matvec=matvec, rmatvec=rmatvec
            )

            if y.ndim < 2:
                out = sparse_lsqr(X_centered, y)
                self.coef_ = out[0]
                self._residues = out[3]
            else:
                # sparse_lstsq cannot handle y with shape (M, K)
                outs = Parallel(n_jobs=n_jobs_)(
                    delayed(sparse_lsqr)(X_centered, y[:, j].ravel())
                    for j in range(y.shape[1])
                )
                self.coef_ = np.vstack([out[0] for out in outs])
                self._residues = np.vstack([out[3] for out in outs])
        else:
            self.coef_, self._residues, self.rank_, self.singular_ = linalg.lstsq(X, y)
            self.coef_ = self.coef_.T

        if y.ndim == 1:
            self.coef_ = np.ravel(self.coef_)
        self._set_intercept(X_offset, y_offset, X_scale)
        return self