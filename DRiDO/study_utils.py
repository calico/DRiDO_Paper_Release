import numpy as np
import os
import pandas as pd
import pickle
import re
import textwrap

from DRiDO.covariance_utils import compute_partial_correlation
from DRiDO.graph_lasso import run_gglasso
from DRiDO.spectral_clustering import make_knn_graph, run_spectral_clustering


ANIMAL_METADATA_COLS = ['MouseID', 'Generation', 'Diet',
                        'Coat', 'SurvDays', 'Died']

DROPPED_META_FEATURES = ['MouseID', 'Meta_PLR',
                         'Meta_AgeRange', 'Meta_AgeInDays']

FACS_NAME_MAPPING = {
    'FLOW_B.CD11BposPercB.AdjBatch': 'FLOW_CD11B+ (% B cells)',
    'FLOW_B.CD62LposPercB.AdjBatch': 'FLOW_CD62L+ (% B cells)',
    'FLOW_B.NKG2DposPercB.AdjBatch': 'FLOW_NKG2D+ (% B cells)',
    'FLOW_B.PercLymph.AdjBatch': 'FLOW_B cells (% lymph)',
    'FLOW_CD4T.CD25posPercCD4T.AdjBatch': 'FLOW_CD25+ (% CD4+ T)',
    'FLOW_CD4T.CD62LnegCD44negPercCD4T.AdjBatch': 'FLOW_CD62L- CD44- (% CD4+ T)',
    'FLOW_CD4T.CD62LnegCD44posPercCD4T.AdjBatch': 'FLOW_CD62L- CD44+ (% CD4+ T)',
    'FLOW_CD4T.CD62LposCD44negPercCD4T.AdjBatch': 'FLOW_CD62L+ CD44- (% CD4+ T)',
    'FLOW_CD4T.CD62LposCD44posPercCD4T.AdjBatch': 'FLOW_CD62L+ CD44+ (% CD4+ T)',
    'FLOW_CD4T.NKG2DposPercCD4T.AdjBatch': 'FLOW_NKG2D+ (% CD4+ T)',
    'FLOW_CD4T.PercT.AdjBatch': 'FLOW_CD4+ (% T)',
    'FLOW_CD8T.CD62LnegCD44negPercCD8T.AdjBatch': 'FLOW_CD62L- CD44- (% CD8+ T)',
    'FLOW_CD8T.CD62LnegCD44posPercCD8T.AdjBatch': 'FLOW_CD62L- CD44+ (% CD8+ T)',
    'FLOW_CD8T.CD62LposCD44negPercCD8T.AdjBatch': 'FLOW_CD62L+ CD44- (% CD8+ T)',
    'FLOW_CD8T.CD62LposCD44posPercCD8T.AdjBatch': 'FLOW_CD62L+ CD44+ (% CD8+ T)',
    'FLOW_CD8T.NKG2DposPercCD8T.AdjBatch': 'FLOW_NKG2D+ (% CD8+ T)',
    'FLOW_CD8T.PercT.AdjBatch': 'FLOW_CD8+ (% T)',
    'FLOW_CLLlike.PercViable.AdjBatch': 'FLOW_CLL-like cells (% viable)',
    'FLOW_DNT.B220posPercDNT.AdjBatch': 'FLOW_B220+ (% double negative T)',
    'FLOW_DNT.CD25posPercDNT.AdjBatch': 'FLOW_CD25+ (% double negative T)',
    'FLOW_DNT.CD62LnegCD44negPercDNT.AdjBatch': 'FLOW_CD62L- CD44- (% double '
    'negative T)',
    'FLOW_DNT.CD62LnegCD44posPercDNT.AdjBatch': 'FLOW_CD62L- CD44+ (% double '
    'negative T)',
    'FLOW_DNT.CD62LposCD44negPercDNT.AdjBatch': 'FLOW_CD62L+ CD44- (% double '
    'negative T)',
    'FLOW_DNT.CD62LposCD44posPercDNT.AdjBatch': 'FLOW_CD62L+ CD44+ (% double '
    'negative T)',
    'FLOW_DNT.NKG2DposPercDNT.AdjBatch': 'FLOW_NKG2D+ (% double negative T)',
    'FLOW_Eos.PercMyeloid.AdjBatch': 'FLOW_Eosinophils (% myeloid)',
    'FLOW_Lymph.PercViable.AdjBatch': 'FLOW_Lymphocytes (% viable)',
    'FLOW_Mono.CD11CposCD62LposPercMono.AdjBatch': 'FLOW_CD11C+ CD62L+ (% mono)',
    'FLOW_Mono.InflPercMono.AdjBatch': 'FLOW_CD11C- CD62L+ (inflamm.) (% mono)',
    'FLOW_Mono.OtherPercMono.AdjBatch': 'FLOW_CD11C- CD62L- (other) (% mono)',
    'FLOW_Mono.PercMyeloid.AdjBatch': 'FLOW_Monocytes (% myeloid)',
    'FLOW_Mono.ResidPercMono.AdjBatch': 'FLOW_CD11C+ CD62L- (resident) (% mono)',
    'FLOW_Myeloid.PercViable.AdjBatch': 'FLOW_Myeloid cells (% viable)',
    'FLOW_NK.B220posPercNK.AdjBatch': 'FLOW_B220+ (% NK)',
    'FLOW_NK.CD11BnegCD11CnegPercCD62LposNK.AdjBatch': 'FLOW_CD11B- CD11C-  (% '
    'CD62L+ NK)',
    'FLOW_NK.CD11BnegCD11CnegPercNK.AdjBatch': 'FLOW_CD11B- CD11C- (% NK)',
    'FLOW_NK.CD11BnegCD11CposPercCD62LposNK.AdjBatch': 'FLOW_CD11B- CD11C+  (% '
    'CD62L+ NK)',
    'FLOW_NK.CD11BnegCD11CposPercNK.AdjBatch': 'FLOW_CD11B- CD11C+ (% NK)',
    'FLOW_NK.CD11BposCD11CnegPercCD62LposNK.AdjBatch': 'FLOW_CD11B+ CD11C-  (% '
    'CD62L+ NK)',
    'FLOW_NK.CD11BposCD11CnegPercNK.AdjBatch': 'FLOW_CD11B+ CD11C- (% NK)',
    'FLOW_NK.CD11BposCD11CposPercCD62LposNK.AdjBatch': 'FLOW_CD11B+ CD11C+  (% '
    'CD62L+ NK)',
    'FLOW_NK.CD11BposCD11CposPercNK.AdjBatch': 'FLOW_CD11B+ CD11C+ (% NK)',
    'FLOW_NK.CD62LposPercNK.AdjBatch': 'FLOW_CD62L+ (% NK)',
    'FLOW_NK.PercLymph.AdjBatch': 'FLOW_Natural killer (NK) cells (% lymph)',
    'FLOW_NKG2DposT.CD4posPercNKG2DposT.AdjBatch': 'FLOW_CD4+ (% NKG2D+ T)',
    'FLOW_NKG2DposT.CD8posPercNKG2DposT.AdjBatch': 'FLOW_CD8+ (% NKG2D+ T)',
    'FLOW_NKG2DposT.DNPercNKG2DposT.AdjBatch': 'FLOW_Double negative (% NKG2D+ T)',
    'FLOW_NKG2DposT.PercT.AdjBatch': 'FLOW_NKG2D+ (% T)',
    'FLOW_Neut.CD62LposPercNeut.AdjBatch': 'FLOW_CD62L+ (% neut)',
    'FLOW_Neut.PercMyeloid.AdjBatch': 'FLOW_Neutrophils (% myeloid)',
    'FLOW_Other.PercViable.AdjBatch': 'FLOW_Other cells (% viable)',
    'FLOW_T.PercLymph.AdjBatch': 'FLOW_T cells (% lymph)'}

def heuristic_label(features):
    feature_to_cluster = {
        'MetCage_EE': 'EE',
        'MetCage_Food': 'Food intake',
        'BW_PhenoDelta': 'BWDelta',
        'MetCage_DeltaVO2': 'MetCage_Delta',
        'MetCage_WheelSpeed': 'Wheel Running',
        'Wheel_Q95SpeedNight': 'Wheel running\n(home cage)',
        'MetCage_PedMeters': 'Activity',
        'FLOW_CD4T.CD62LnegCD44posPercCD4T': 'CD4T',
        'FLOW_Lymph.PercViable': 'Lymphocytes',
        'FLOW_Eos.PercMyeloid': 'Eosinophils',
        'FLOW_NKG2DposT.CD8posPercNKG2DposT': 'NKG2D+ T cells',
        'FLOW_NK.CD11BposCD11CnegPercNK': 'Mature NK cells',
        'FLOW_NK.CD62LposPercNK': 'CD62L+ NK cells',
        'CBC_RDWcv': 'RDW',
        'CBC_Hgb': 'Hemoglobin',
        'CBC_NLR': 'NLR',
        'PIXI_PercFat': 'Adiposity',
        'Echo_EF': 'Ejection frac',
        'Void_logTotalVol': 'Voiding',
        'Frailty_FrailtyAdj': 'Frailty',
        'Frailty_Cataracts': 'Eye',
        'Glu.F_Glucose': 'Glucose',
        'AS_SlopeLog': 'Acoustic startle',
        'Meta_DaysToDeath': 'TTD',
        'Meta_Lifespan': 'Lifespan',
        'Meta_timepoint': 'Age',
        'Meta_DietGroup': 'Diet',
        'Meta_CAST': 'CAST'
    }
    priority_classes = ['Body composition', 'Glucose']
    classes = [cls for feat, cls in feature_to_cluster.items()
               if (feat in features.values)]
    classes = sorted(classes, key=lambda x: 0 if x in priority_classes else 1)
    name = ' /\n'.join(classes)
    # name = "\n".join(textwrap.wrap(name, width=32))
    if len(classes) == 0:
        # Just pick the first feature whatever that is
        name = features.values[0]

    return name


def process_one_year(wide_df, year):
    prefix = f'Y{year}'

    # Grab subset of columns for this year
    cols = ['MouseID'] + [x for x in wide_df.columns if x.startswith(prefix)]
    subset_df = wide_df[cols].set_index('MouseID').copy()
    mask = subset_df.var() > 1e-5
    subset_df = subset_df.loc[:, mask]

    # We might have multiple submeasurements so combine those into one by averaging
    year_prefix = re.compile(f'Y{year}[A-E]*_')
    subset_df.columns = [year_prefix.sub('', x) for x in subset_df.columns]

    # Each phenotyping event has an age associated, we're going to take the median
    # age of phenotyping as our "age". We also take the range to make sure
    # that there isn't anything wrong with the assignment of measurement to years
    age_in_days = re.compile(r'[^_]+_AgeInDays')
    subset_df.columns = [age_in_days.sub(
        'Age_AgeInDays', x) for x in subset_df.columns]

    # Take the mean
    agg_df = subset_df.groupby(subset_df.columns, axis=1).agg(np.nanmean)
    agg_df.loc[:, 'Meta_AgeInDays'] = subset_df.loc[:,
                                                    subset_df.columns == 'Age_AgeInDays'].median(axis=1)
    agg_df = agg_df.drop(columns=['Age_AgeInDays'])
    agg_df.loc[:, 'Meta_AgeRange'] = subset_df.loc[:, subset_df.columns == 'Age_AgeInDays'].max(
        axis=1) - subset_df.loc[:, subset_df.columns == 'Age_AgeInDays'].min(axis=1)
    agg_df.loc[:, 'Meta_timepoint'] = year
    return agg_df


def make_collapsed_df(df, animal_metadata):
    diet_group2index = {'AL': 0, '1D': 1, '20': 1, '2D': 1, '40': 2}
    combined_dfs = []
    columns = set()

    for year in [1, 2, 3]:
        year_df = process_one_year(df, year)
        diet_idx = [diet_group2index[x.split('-')[1]] for x in year_df.index]
        year_df.loc[:, 'Meta_DietGroup'] = 0 if year == 1 else diet_idx
        year_df = year_df.reset_index()
        if len(columns) == 0:
            columns = set(year_df.columns)
        else:
            columns = columns & set(year_df.columns)
        combined_dfs.append(year_df)
    columns = list(columns)
    combined_df = pd.concat([df[columns] for df in combined_dfs], axis=0)

    # Compute survival covariates
    surv_days = animal_metadata.set_index(
        "MouseID")["SurvDays"].loc[combined_df['MouseID']]
    combined_df.loc[:, 'Meta_DaysToDeath'] = (
        surv_days.values - combined_df['Meta_AgeInDays'].values)
    is_dead = animal_metadata.set_index(
        "MouseID")["Died"].loc[combined_df['MouseID']].values
    combined_df.loc[~is_dead, 'Meta_DaysToDeath'] = None
    combined_df.loc[:, 'Meta_PLR'] = (
        surv_days.values - combined_df['Meta_AgeInDays'].values) / surv_days.values
    combined_df.loc[~is_dead, 'Meta_PLR'] = None

    # Drop BWTest columns
    columns_to_drop = [x for x in combined_df.columns if x.endswith('BWTest')]
    combined_df = combined_df.drop(columns=columns_to_drop)

    missing_mask = combined_df['Meta_AgeInDays'].isnull()
    combined_df = combined_df[~missing_mask]
    return combined_df


def load_data(csv_filename):
    with open(csv_filename, 'r') as f:
        wide_df = pd.read_csv(f)
    animal_metadata = wide_df[ANIMAL_METADATA_COLS]
    data = make_collapsed_df(wide_df, animal_metadata)
    return data, animal_metadata


def make_arg_str(**kwargs):
    arg_str = '_'.join([f'{k}={v}' for k, v in kwargs.items()])
    return arg_str


def name_clusters(cluster_df, split_meta_clusters=True):
    if split_meta_clusters:
        cluster_df.loc[cluster_df['var'] == 'Meta_AgeInDays', 'cluster'] = -1
        cluster_df.loc[cluster_df['var'] == 'Meta_timepoint', 'cluster'] = -1
        cluster_df.loc[cluster_df['var'] == 'Meta_DaysToDeath', 'cluster'] = -2
        cluster_df.loc[cluster_df['var'] == 'Meta_Lifespan', 'cluster'] = -2
        cluster_df.loc[cluster_df['var'] == 'Meta_DietGroup', 'cluster'] = -3
        cluster_df.loc[cluster_df['var'] == 'Meta_CAST', 'cluster'] = -4
        cluster_df.loc[cluster_df['var'] == 'BW_PhenoDelta', 'cluster'] = -5

    cluster_map = cluster_df.groupby('cluster').agg(
        heuristic_label).to_dict()['var']
    cluster_df.loc[:, 'cluster'] = (cluster_df.loc[:, 'cluster']
                                              .replace(cluster_map))
    return cluster_df


def compute_glasso_network(
        data,
        max_glasso_lambda,
        use_latent,
        use_scaling,
        cluster_knn_sparsity,
        use_partial_corr,
        split_meta_clusters,
        force_recompute,
        num_clusters=20,
        fix_lambda=None,
        prefix=None,
):
    if prefix is None:
        prefix = 'v3'

    arg_str = make_arg_str(
        lmbda=max_glasso_lambda,
        latent=use_latent,
        scaling=use_scaling,
        k=cluster_knn_sparsity,
        partial_corr=use_partial_corr)
    cache_path = f'cache/cached_glasso_result_{prefix}_{arg_str}'

    features_to_drop = [x for x in DROPPED_META_FEATURES if x in data.columns]
    data = data.drop(columns=features_to_drop).copy()

    if os.path.exists(cache_path) and not force_recompute:
        print('USING CACHE')
        with open(cache_path, 'rb') as f:
            result = pickle.load(f)
            if len(result) == 3:
                theta, k_sparse_abs_theta, cluster_df = result
                partial_correlation = compute_partial_correlation(
                    theta, should_remove_diag=False)
            else:
                theta, partial_correlation, k_sparse_abs_theta, cluster_df = result
    else:
        # Recompute theta etc.
        log_lambda = np.log10(max_glasso_lambda)
        lambda_range = np.logspace(0, log_lambda, 10)
        print(lambda_range)
        mu_range = np.logspace(100.0, 50.0, 10)
        theta = run_gglasso(data, mu_range, lambda_range,
                            fix_lambda=fix_lambda, use_latent=use_latent, do_scaling=use_scaling)
        partial_correlation = compute_partial_correlation(
            theta, should_remove_diag=False)
        if use_partial_corr:
            k_sparse_abs_theta = make_knn_graph(
                partial_correlation, k=cluster_knn_sparsity)
            cluster_df = run_spectral_clustering(
                k_sparse_abs_theta, k=num_clusters)
        else:
            k_sparse_abs_theta = make_knn_graph(theta, k=cluster_knn_sparsity)
            cluster_df = run_spectral_clustering(
                k_sparse_abs_theta, k=num_clusters)

        with open(cache_path, 'wb') as f:
            pickle.dump((theta, partial_correlation,
                         k_sparse_abs_theta, cluster_df), f)

    cluster_df = name_clusters(
        cluster_df, split_meta_clusters=split_meta_clusters)
    return theta, partial_correlation, k_sparse_abs_theta, cluster_df
