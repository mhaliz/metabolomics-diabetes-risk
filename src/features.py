import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# Cross validation function of feature selection pipeline

def cv_feature_selection(pipeline, X, y, biomarker_cols, non_biomarker_cols, n_splits=5, random_state=33):
    """
    Runs cross-validated feature selection and returns:
    - per-fold summary tables
    - aggregate stability summary across folds
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_summaries = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_fold_train = X.iloc[train_idx]
        y_fold_train = y.iloc[train_idx]

        # Fit pipeline on this fold's training data
        pipeline.fit(X_fold_train, y_fold_train)

        # Reconstruct feature names
        preprocessor = pipeline.named_steps['preprocess']
        sel = pipeline.named_steps['select']

        var_step = preprocessor.named_transformers_['biomarkers'].named_steps['variance']
        genes_after_var = np.array(biomarker_cols)[var_step.get_support()]
        combined_names = np.array(list(genes_after_var) + non_biomarker_cols)

        # Chain masks
        cor_mask = ~np.isin(np.arange(len(combined_names)),
                                   sel.named_steps['correlation'].drop_cols_)
        after_cor = combined_names[cor_mask]

        uni_mask = sel.named_steps['univariate'].get_support()
        after_uni = after_cor[uni_mask]

        las_mask = sel.named_steps['lasso'].get_support()
        final_features = after_uni[las_mask]

        # Collect statistics
        variance_scores = pd.Series(var_step.variances_, index=biomarker_cols)

        univariate_scores = pd.DataFrame({
            'f_score': sel.named_steps['univariate'].scores_,
            'p_value': sel.named_steps['univariate'].pvalues_,
        }, index=after_cor)

        lasso_coefs = pd.Series(
            sel.named_steps['lasso'].estimator_.coef_.flatten(),
            index=after_uni
        )

        # Build fold summary table
        summary = pd.DataFrame(index=final_features)
        summary['fold'] = fold
        summary['variance'] = variance_scores.reindex(final_features)
        summary['f_score'] = univariate_scores['f_score'].reindex(final_features)
        summary['p_value'] = univariate_scores['p_value'].reindex(final_features)
        summary['lasso_coef'] = lasso_coefs.reindex(final_features)
        summary['abs_coef'] = summary['lasso_coef'].abs()
        summary.index.name = 'feature'
        summary = summary.sort_values('abs_coef', ascending=False)

        fold_summaries.append(summary)

        print(f"Fold {fold}: {len(final_features)} features selected → {list(final_features)}")

    # Aggregate stability summary across folds
    all_folds = pd.concat(fold_summaries).reset_index()

    stability = (all_folds
        .groupby('feature')
        .agg(
            times_selected = ('fold', 'count'),       # how many folds it appeared in
            mean_f_score = ('f_score', 'mean'),
            std_f_score = ('f_score', 'std'),
            mean_lasso_coef = ('lasso_coef', 'mean'),
            std_lasso_coef = ('lasso_coef', 'std'),
            mean_abs_coef = ('abs_coef', 'mean'),
            mean_variance = ('variance', 'mean'),
        )
        .sort_values(['times_selected', 'mean_abs_coef'], ascending=[False, False])
    )
    stability['selected_pct'] = (stability['times_selected'] / n_splits * 100).astype(int)

    return fold_summaries, stability