import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import argparse
import warnings
import pandas as pd
import numpy as np
import json
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")

# Import the same utility functions from the extended linear model
from src.linear.utils import set_random_seed, ancestry_encoding
from src.linear.utils import logo_splits, sk_splits, stratified_k_fold_splits, validation_split, create_dir_if_not_exists

def load_features(data_path, dataset, sample_ids):
    """Load feature data for given sample IDs."""
    features_list = []
    for sample_id in sample_ids:
        # Adjust this path based on your actual data structure
        feature_file = f'{data_path}/{dataset}/feats/{sample_id}.npy'
        features = np.load(feature_file)
        features_list.append(features)
    return np.array(features_list)

def compute_avg_features(features, feature_indices, covariates=None, n_genes=17759, n_features_per_gene=78):
    """Compute average of specified feature indices for each gene.
    
    Features shape: (n_samples, n_genes * n_features_per_gene)
    We reshape to (n_samples, n_genes, n_features_per_gene), extract the specified 
    feature indices from each gene, and average them per gene.
    
    Returns shape: (n_samples, n_genes + n_covariates)
    """
    n_samples = features.shape[0]
    
    # Reshape to (n_samples, n_genes, n_features_per_gene)
    features_reshaped = features.reshape(n_samples, n_genes, n_features_per_gene)
    
    # Extract the specified feature indices for ALL genes
    # Shape: (n_samples, n_genes, len(feature_indices))
    selected_features = features_reshaped[:, :, feature_indices]
    
    # Average across the feature indices dimension (axis=2) to get one value per gene
    # Result shape: (n_samples, n_genes) = (n_samples, 17759)
    avg_features = np.mean(selected_features, axis=2)
    
    if covariates is not None and covariates.shape[1] > 0:
        avg_features = np.concatenate([avg_features, covariates], axis=1)
    
    return avg_features

def evaluate_split(train_features, train_labels, train_covariates,
                   val_features, val_labels, val_covariates,
                   test_features, test_labels, test_covariates,
                   feature_indices, C=1.0, random_state=42):
    """Train logistic regression on averaged features and evaluate."""
    
    # Compute averaged features with optional covariates
    X_train = compute_avg_features(train_features, feature_indices, train_covariates)
    X_val = compute_avg_features(val_features, feature_indices, val_covariates)
    X_test = compute_avg_features(test_features, feature_indices, test_covariates)
    
    # Train logistic regression model
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, penalty='l1', solver='liblinear', max_iter=1000, random_state=random_state)
    )
    model.fit(X_train, train_labels)
    
    # Get predictions (probabilities)
    train_probs = model.predict_proba(X_train)[:, 1]
    val_probs = model.predict_proba(X_val)[:, 1]
    test_probs = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    train_auroc = roc_auc_score(train_labels, train_probs)
    train_auprc = average_precision_score(train_labels, train_probs)
    
    val_auroc = roc_auc_score(val_labels, val_probs)
    val_auprc = average_precision_score(val_labels, val_probs)
    
    test_auroc = roc_auc_score(test_labels, test_probs)
    test_auprc = average_precision_score(test_labels, test_probs)
    
    # Get model coefficients
    coefs = model.named_steps['logisticregression'].coef_[0]
    intercept = model.named_steps['logisticregression'].intercept_[0]
    
    return {
        'train': {'auroc': train_auroc, 'auprc': train_auprc, 'probs': train_probs},
        'val': {'auroc': val_auroc, 'auprc': val_auprc, 'probs': val_probs},
        'test': {'auroc': test_auroc, 'auprc': test_auprc, 'probs': test_probs},
        'model': {'coefs': coefs, 'intercept': intercept}
    }

def store_predictions(predictions_dict, output_dir, file_name):
    """Store predictions to CSV."""
    df = pd.DataFrame.from_dict(predictions_dict, orient='index')
    df.to_csv(os.path.join(output_dir, file_name))

def store_coefficients(coefs, gene_to_index, covariate_cols, output_dir, file_name):
    """Store model coefficients with gene names to CSV."""
    # Create reverse mapping: index -> gene name
    # gene_to_index has STRING keys like {"0": "SAMD11", "1": "NOC2L", ...}
    index_to_gene = {int(idx): gene_name for idx, gene_name in gene_to_index.items()}
    
    # Number of gene features (based on coefficients, not gene_to_index)
    n_gene_coefs = len(coefs) - (len(covariate_cols) if covariate_cols else 0)
    n_covariates = len(covariate_cols) if covariate_cols else 0
    
    coef_data = []
    
    # Add gene coefficients
    for i in range(n_gene_coefs):
        gene_name = index_to_gene.get(i, f"gene_index_{i}")  # Fallback if gene not found
        coef_data.append({
            'feature_type': 'gene',
            'feature_name': gene_name,
            'gene_index': i,
            'coefficient': coefs[i]
        })
    
    # Add covariate coefficients
    if n_covariates > 0 and covariate_cols is not None:
        for i, cov_name in enumerate(covariate_cols):
            coef_data.append({
                'feature_type': 'covariate',
                'feature_name': cov_name,
                'gene_index': -1,
                'coefficient': coefs[n_gene_coefs + i]
            })
    
    df = pd.DataFrame(coef_data)
    
    # Sort by absolute coefficient value
    df['abs_coef'] = df['coefficient'].abs()
    df = df.sort_values('abs_coef', ascending=False).drop('abs_coef', axis=1)
    
    df.to_csv(os.path.join(output_dir, file_name), index=False)
    
    # Print summary
    non_zero = (df['coefficient'] != 0).sum()
    print(f"  Saved {len(df)} coefficients ({non_zero} non-zero) to {file_name}")
    
    return df

def parse_args():
    parser = argparse.ArgumentParser(description="Logistic Regression on Average Features")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--feature_indices", type=str, default="9,11,13", 
                        help="Comma-separated feature indices to average (e.g., '0,5,10')")
    parser.add_argument("--C", type=float, default=(1/1.5), 
                        help="Inverse regularization strength for LogisticRegression")
    parser.add_argument("--exclude", type=str, default=None)
    parser.add_argument("--cohort", type=str, default="full")
    parser.add_argument("--logo", action="store_true")
    parser.add_argument("--stratified_kfold", action="store_true")
    parser.add_argument("--tts", action="store_true")
    parser.add_argument("--test_group", type=str, default=None)
    parser.add_argument("--shuffle_controls", action="store_true")
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--silent", action="store_true")
    parser.add_argument("-rs", "--random_state", type=int, default=42)
    parser.add_argument("-o", "--output", type=str, default=".")
    parser.add_argument('--pop', type=str, default=None, help='Which population to filter for (eg. EUR, EAS, etc.)')
    parser.add_argument('--pop_file', type=str, default=None, help='Population file')
    parser.add_argument('--pop_threshold', type=float, help='Population threshold', default=0.85)
    parser.add_argument('--covariates', type=str, default=None, help='File specifying the covariates')
    parser.add_argument('--af', type=float, default=1.0, help='Allele frequency threshold')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    set_random_seed(args.random_state)
    
    # Parse feature indices
    feature_indices = [int(x) for x in args.feature_indices.split(',')]
    print(f"Using feature indices: {feature_indices}")
    print(f"Regularization C: {args.C}")
    
    # Load info DataFrame
    info_df = pd.read_csv(f'{args.data_path}/{args.dataset}/info.csv')
    non_pop_samples = None
    
    ## Add covariates if we should
    n_covariates = 0
    covariates = np.empty((info_df.shape[0], 0))
    if args.covariates is not None:
        covariates_df = pd.read_csv(args.covariates)
        n_covariates = covariates_df.shape[1] - 1
        info_df = pd.merge(info_df, covariates_df, how='left', 
                        left_on='sample_id', right_on='sample_id')
        covariate_cols = list(info_df.columns[-n_covariates:])
        print(f"Including covariates: {covariate_cols}")
    
    ## POP filter if we should
    if args.pop is not None:
        assert args.pop_file is not None, "Error: --pop_file must be specified if --pop is provided."
        print(f"Will begin pop-filter of {info_df.shape[0]} samples for {args.pop} ≥ {args.pop_threshold}")
        pop = pd.read_csv(args.pop_file)
        pop.loc[:, 'IID'] = pop['FID'].astype(str).str.cat(pop['SID'].astype(str), sep='_')
        pop = pop.set_index('IID', drop=True)
        pop_filter = pop[pop[args.pop].astype(float) >= args.pop_threshold].copy()

        n_original_samples = info_df.shape[0]
        non_pop_indices = pop.index.difference(pop_filter.index).intersection(info_df['sample_id'])
        pop_indices = pop_filter.index.intersection(info_df['sample_id'])
        non_pop_samples = info_df[info_df['sample_id'].isin(non_pop_indices)]
        info_df = info_df[info_df['sample_id'].isin(pop_indices)]
        
        
        # Double check
        non_pop_double_check = pd.merge(non_pop_samples.set_index('sample_id'), pop, left_index=True, right_index=True)
        assert non_pop_double_check[args.pop].max() < args.pop_threshold, f"Non pop samples were not correctly filtered {non_pop_double_check[args.pop].max()} > {args.pop_threshold}"
        print(f"Population filtered {n_original_samples} -> {info_df.shape[0]} samples remain, {non_pop_samples.shape[0]} filtered out")
    
    # Read gene to index (for compatibility, even though we may not use it)
    json_data = open(f'{args.data_path}/gene_to_index_{args.af}.json', 'r')
    gene_to_index = json.load(json_data)
    
    # Set the 'group' column based on the first character of 'sample_id'
    info_df['group'] = info_df['sample_id'].str[0]
    if args.pop is not None and non_pop_samples is not None:
        non_pop_samples['group'] = non_pop_samples['sample_id'].str[0]
    
    # Reassign 'I' cases to 'E'
    info_df.loc[(info_df['group'] == 'I') & (info_df['label'] == 1), 'group'] = 'E'
    
    if args.exclude is not None:
        exclude_drop = info_df[info_df['group'] == args.exclude]
        info_df = info_df.drop(exclude_drop.index)

        print(f"Excluded {exclude_drop.shape[0]} samples from cohort {args.exclude}")
    
    if args.shuffle_controls:
        # Remove control samples from the original DataFrame to avoid duplication
        case_df = info_df[info_df['label'] == 1]
        case_cohorts = case_df['group'].unique().tolist()

        # Shuffle and handle controls separately
        controls_df = info_df[info_df['label'] == 0].sample(frac=1, random_state=args.random_state).reset_index(drop=True)

        # Calculate the total number of cases in E, F, U for proportion
        total_cases = case_df['group'].isin(case_cohorts).sum()
        total_controls = controls_df.shape[0]

        # Assign controls to case cohorts E, F, U proportionally
        for cohort in case_cohorts:
            cohort_cases = case_df[case_df['group'] == cohort].shape[0]
            proportion_controls = int(total_controls * (cohort_cases / total_cases))
            assigned_controls = controls_df[:proportion_controls]
            controls_df = controls_df[proportion_controls:]
            assigned_controls['group'] = cohort
            case_df = pd.concat([case_df, assigned_controls])

        # Handle any remaining controls by assigning them to the groups with the highest case counts
        while not controls_df.empty:
            for cohort in case_cohorts:
                if controls_df.empty:
                    break
                control = controls_df.iloc[0:1]
                control['group'] = cohort
                case_df = pd.concat([case_df, control])
                controls_df = controls_df.iloc[1:].reset_index(drop=True)

        info_df = case_df
    
    if args.cohort != "full":
        info_df = info_df[~((info_df['group'] != args.cohort) & (info_df['label'] == 1))]
        if args.bootstrap and not args.logo:
            bootstrap_controls = info_df[info_df['label'] == 0].sample(frac=1, random_state=args.random_state).reset_index(drop=True)
            info_df = info_df[info_df['label'] == 1]
            info_df = pd.concat([info_df, bootstrap_controls.head(info_df.shape[0])])
    
    # Reset index to ensure continuity
    info_df = info_df.reset_index(drop=True)

    assert info_df['sample_id'].nunique() == len(info_df), "Duplicated sample IDs found."
    
    sample_ids = info_df['sample_id'].values
    labels = info_df['label'].values
    groups = info_df['group'].values
    
    # Generate splits (same as extended linear model)
    if args.logo:
        splits = logo_splits(labels, groups, random_state=args.random_state)
    elif args.stratified_kfold:
        if args.test_group is not None:
            if args.test_group == "NON_POP":
                sample_ids = np.concatenate([sample_ids, non_pop_samples['sample_id'].values])
                labels = np.concatenate([labels, non_pop_samples['label'].values])
                info_df = pd.concat([info_df, non_pop_samples]).reset_index(drop=True)
                test_indices = info_df.index.values[-non_pop_samples.shape[0]:]
            else:
                test_indices = info_df[info_df['group'] == args.test_group].index.values
            splits = stratified_k_fold_splits(labels, n_splits=5, random_state=args.random_state, 
                                            test_indices=test_indices, test_group=args.test_group)
        else:
            splits = stratified_k_fold_splits(labels, random_state=args.random_state)
    elif args.tts:
        if args.test_group is not None:
            if args.test_group == "NON_POP":
                sample_ids = np.concatenate([sample_ids, non_pop_samples['sample_id'].values])
                labels = np.concatenate([labels, non_pop_samples['label'].values])
                info_df = pd.concat([info_df, non_pop_samples]).reset_index(drop=True)
                test_indices = info_df.index.values[-non_pop_samples.shape[0]:]
            else:
                test_indices = info_df[info_df['group'] == args.test_group].index.values
            splits = validation_split(labels, random_state=args.random_state, 
                                    test_indices=test_indices, test_group=args.test_group)
        else:
            splits = validation_split(labels, random_state=args.random_state)
    else:
        splits = sk_splits(labels, random_state=args.random_state)

    # NOW extract covariates
    if n_covariates > 0:
        covariates = info_df[covariate_cols].values.astype(np.float32)
    else:
        covariates = np.empty((info_df.shape[0], 0))
    
    # Output file
    filename = f"{args.output}/logreg_avg_feature_{args.cohort}cohort_{args.shuffle_controls}shufflecontrols_{args.bootstrap}bootstrap_{args.logo}logo_{args.af}af_exc{args.exclude}.csv"
    
    # Process each split
    all_results = []
    for split_id, (train_ids, val_ids, test_ids, train_groups, val_groups, test_groups) in enumerate(splits):

        first_sample_features = np.load(f'{args.data_path}/{args.dataset}/feats/{sample_ids[0]}.npy')
        print(f"\n=== SANITY CHECK ===")
        print(f"First sample ID: {sample_ids[0]}")
        print(f"First sample features shape: {first_sample_features.shape}")
        print(f"First sample features: {first_sample_features[:20]}")  # Show first 20 values
        print(f"====================\n")
        
        # Assertions for no overlap
        assert len(set(train_ids) & set(val_ids)) == 0, "Overlap found between training and validation sets"
        assert len(set(train_ids) & set(test_ids)) == 0, "Overlap found between training and test sets"
        assert len(set(val_ids) & set(test_ids)) == 0, "Overlap found between validation and test sets"
        
        print(f"\n\nSplit {split_id} --> Training groups: {train_groups} | Validation groups: {val_groups} | Test groups: {test_groups}")
        
        # Get case/control counts
        case_control_counts_train = info_df.iloc[train_ids, :].groupby(['group', 'label']).size().unstack(fill_value=0)
        case_control_counts_val = info_df.iloc[val_ids, :].groupby(['group', 'label']).size().unstack(fill_value=0)
        case_control_counts_test = info_df.iloc[test_ids, :].groupby(['group', 'label']).size().unstack(fill_value=0)
        print("\nCase and Control counts per group:")
        print("TRAIN:")
        print(case_control_counts_train)
        print("VAL:")
        print(case_control_counts_val)
        print("TEST:")
        print(case_control_counts_test)
        
        # Base AUPRC (prevalence)
        base_auprc_train = (labels[train_ids] == 1).sum() / labels[train_ids].shape[0]
        base_auprc_val = (labels[val_ids] == 1).sum() / labels[val_ids].shape[0]
        base_auprc_test = (labels[test_ids] == 1).sum() / labels[test_ids].shape[0]
        
        # Load features for this split
        train_features = load_features(args.data_path, args.dataset, sample_ids[train_ids])
        val_features = load_features(args.data_path, args.dataset, sample_ids[val_ids])
        test_features = load_features(args.data_path, args.dataset, sample_ids[test_ids])
        
        # Get covariates for this split
        train_covariates = covariates[train_ids] if n_covariates > 0 else None
        val_covariates = covariates[val_ids] if n_covariates > 0 else None
        test_covariates = covariates[test_ids] if n_covariates > 0 else None
        
        # Train logistic regression and evaluate
        results = evaluate_split(
            train_features, labels[train_ids], train_covariates,
            val_features, labels[val_ids], val_covariates,
            test_features, labels[test_ids], test_covariates,
            feature_indices,
            C=args.C,
            random_state=args.random_state
        )
        
        print(f"----------------Split {split_id} final result----------------")
        print(f"Training groups: {train_groups} | Validation groups: {val_groups} | Test groups: {test_groups}")
        print(f"Train - AUROC: {results['train']['auroc']:.4f}, AUPRC: {results['train']['auprc']:.4f}")
        print(f"Val   - AUROC: {results['val']['auroc']:.4f}, AUPRC: {results['val']['auprc']:.4f}")
        print(f"Test  - AUROC: {results['test']['auroc']:.4f}, AUPRC: {results['test']['auprc']:.4f}")
        print(f"Model - Coefs: {results['model']['coefs']}, Intercept: {results['model']['intercept']:.4f}")
        
        if not args.silent:
            # Create output directories
            create_dir_if_not_exists(args.output)
            create_dir_if_not_exists(f"{args.output}/val_predictions")
            create_dir_if_not_exists(f"{args.output}/test_predictions")
            create_dir_if_not_exists(f"{args.output}/train_predictions")
            create_dir_if_not_exists(f"{args.output}/coefficients")
            
            # Store predictions
            train_predictions = {sample_ids[train_ids][i]: results['train']['probs'][i] 
                               for i in range(len(train_ids))}
            val_predictions = {sample_ids[val_ids][i]: results['val']['probs'][i] 
                             for i in range(len(val_ids))}
            test_predictions = {sample_ids[test_ids][i]: results['test']['probs'][i] 
                              for i in range(len(test_ids))}
            
            store_predictions(val_predictions, f"{args.output}/val_predictions", 
                            f"{args.random_state}rs_split{split_id}_val_predictions.csv")
            store_predictions(test_predictions, f"{args.output}/test_predictions", 
                            f"{args.random_state}rs_split{split_id}_test_predictions.csv")
            store_predictions(train_predictions, f"{args.output}/train_predictions", 
                            f"{args.random_state}rs_split{split_id}_train_predictions.csv")

            coef_cols = covariate_cols if n_covariates > 0 else None
            store_coefficients(results['model']['coefs'], gene_to_index, coef_cols,
                            f"{args.output}/coefficients",
                            f"{args.random_state}rs_split{split_id}_coefficients.csv")
        
        # Store results
        result_row = {
            "Split ID": split_id,
            "Random State": args.random_state,
            "Train groups": str(train_groups),
            "Validation groups": str(val_groups),
            "Test groups": str(test_groups),
            "Base AUPRC": f"Train: {base_auprc_train:.4f} | Val: {base_auprc_val:.4f} | Test: {base_auprc_test:.4f}",
            "Feature Indices": str(feature_indices),
            "C": args.C,
            "N Covariates": n_covariates,
            "Train (auroc)": results['train']['auroc'],
            "Train (auprc)": results['train']['auprc'],
            "Validation (auroc)": results['val']['auroc'],
            "Validation (auprc)": results['val']['auprc'],
            "Test (auroc)": results['test']['auroc'],
            "Test (auprc)": results['test']['auprc'],
            "Model Intercept": results['model']['intercept'],
            "N Non-Zero Coefs": np.count_nonzero(results['model']['coefs'])  # Useful for L1
        }
        
        # Add individual coefficients
        for i, coef in enumerate(results['model']['coefs']):
            if i == 0:
                result_row["Avg Feature Coef"] = coef
            else:
                result_row[f"Covariate {i} Coef"] = coef
        
        all_results.append(result_row)
    
    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(filename, index=False)
    print(f"\nResults saved to {filename}")
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Mean Test AUROC: {results_df['Test (auroc)'].mean():.4f} ± {results_df['Test (auroc)'].std():.4f}")
    print(f"Mean Test AUPRC: {results_df['Test (auprc)'].mean():.4f} ± {results_df['Test (auprc)'].std():.4f}")
