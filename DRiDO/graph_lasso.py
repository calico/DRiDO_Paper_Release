
import numpy as np
import pandas as pd

from gglasso.problem import glasso_problem
from DRiDO.covariance_utils import make_npn_corr


def run_gglasso(feature_df, mu_range, lambda_range, fix_lambda=None, use_latent=True, do_scaling=True):
    if fix_lambda:
        lambda_range = np.array([fix_lambda])
    raw_npn_cov = make_npn_corr(
        feature_df, ensure_spsd=True, diag_exaggeration=1.0)
    N = feature_df.shape[0]
    problem = glasso_problem(raw_npn_cov.values, N,
                             reg_params={'lambda1': 0.05, 'mu1': 1000},
                             latent=use_latent, do_scaling=do_scaling)
    model_select_params = {
        'lambda1_range': lambda_range, 'mu1_range': mu_range}
    problem.model_selection(modelselect_params=model_select_params,
                            method='eBIC', gamma=0.1)
    theta = problem.solution.precision_
    theta = pd.DataFrame(theta, index=raw_npn_cov.index,
                         columns=raw_npn_cov.columns)
    sparsity = np.mean(np.abs(theta.values) > 0)

    print(problem.modelselect_stats['BEST'])
    print(sparsity)
    return theta
