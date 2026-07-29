import pandas as pd
import numpy as np
from scipy import stats
import sys

def main():
    try:
        df_test = pd.read_csv('/mnt/c/work/profiler_testing/dl_model/v4_test_errors_detailed.csv')
        df_val = pd.read_csv('/mnt/c/work/profiler_testing/dl_model/validation_errors_detailed.csv')
        df = pd.concat([df_test, df_val], ignore_index=True)
        
        # We only care about FP and FN (these files should only contain errors, but just in case)
        df = df[df['error_type'].isin(['FP', 'FN'])]
        
        # Replace infinite values with NaN
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        fp = df[df['error_type'] == 'FP']
        fn = df[df['error_type'] == 'FN']
        
        features = [
            'model_probability', 'exon_number', 'kmer4_cosine', 'transition_ratio', 'cai_ratio', 
            'gc_diff_overall', 'enc_diff', 'rscu_distance', 'gc3_diff', 'four_fold_id', 
            'grantham_mean', 'grantham_max', 'orf_dominance_diff', 'ka', 'ks', 'kaks_ratio', 
            'kaks_window_variance', 'pct_intact_orf', 'nonsyn_spatial_entropy', 
            'polarity_preserving_frac', 'cpg_tpg_rate', 'period3_power', 'spectral_entropy', 
            'exon_position_relative', 'align_longest_ungapped_norm', 'pct_cds_retained', 
            'aa_freq_cosine', 'n_valid_codons'
        ]
        
        print(f"Total FP: {len(fp)}")
        print(f"Total FN: {len(fn)}")
        print("-" * 50)
        print(f"{'Feature':<30} | {'FP Median':<12} | {'FN Median':<12} | {'p-value':<12} | {'Direction'}")
        print("-" * 80)
        
        results = []
        for feat in features:
            if feat not in df.columns:
                continue
            
            fp_vals = fp[feat].dropna()
            fn_vals = fn[feat].dropna()
            
            if len(fp_vals) == 0 or len(fn_vals) == 0:
                continue
                
            fp_med = fp_vals.median()
            fn_med = fn_vals.median()
            
            # Mann-Whitney U test
            stat, pval = stats.mannwhitneyu(fp_vals, fn_vals, alternative='two-sided')
            
            direction = "FP > FN" if fp_med > fn_med else "FN > FP"
            if pval > 0.05:
                direction = "NS"
                
            results.append({
                'feature': feat,
                'fp_med': fp_med,
                'fn_med': fn_med,
                'pval': pval,
                'direction': direction
            })
            
        # Sort by p-value
        results.sort(key=lambda x: x['pval'])
        for r in results:
            print(f"{r['feature']:<30} | {r['fp_med']:<12.4f} | {r['fn_med']:<12.4f} | {r['pval']:<12.2e} | {r['direction']}")
            
        print("\n\n--- SPECIES DISTRIBUTION ---")
        print("False Positives:")
        print(fp['species'].value_counts(normalize=True).head(10))
        print("\nFalse Negatives:")
        print(fn['species'].value_counts(normalize=True).head(10))
        
        print("\n\n--- NULL/NAN VALUES ANALYSIS (Top 10 features with most missing values) ---")
        missing = df.isnull().mean() * 100
        print(missing.sort_values(ascending=False).head(10))

    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
