#!pip install pandas numpy networkx statsmodels scipy

import pandas as pd
import numpy as np
import networkx as nx
from statsmodels.tsa.stattools import grangercausalitytests
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.multitest import multipletests
import warnings
from google.colab import files
warnings.filterwarnings('ignore')

def check_granger_causality(x, y, max_lag=2, f_stat_threshold=10, p_value_threshold=0.05):
    """Test if x Granger-causes y."""
    try:
        # Pre-filter for insufficient data
        if len(x) < max_lag + 10:
            return False, 0, 1, 0

        # Stack arrays directly instead of using DataFrame
        xy_data = np.column_stack([y, x])

        # Remove rows with NaN
        mask = ~np.isnan(xy_data).any(axis=1)
        xy_data = xy_data[mask]

        if len(xy_data) < max_lag + 10:
            return False, 0, 1, 0

        test_result = grangercausalitytests(
            xy_data,
            maxlag=max_lag,
            verbose=False)

        f_stats = [test_result[lag][0]['ssr_ftest'][0] for lag in range(1, max_lag + 1)]
        p_values = [test_result[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag + 1)]

        max_f_index = np.argmax(f_stats)
        best_f = f_stats[max_f_index]
        best_p = p_values[max_f_index]
        best_lag = max_f_index + 1

        is_causal = (best_f > f_stat_threshold) and (best_p < p_value_threshold)

        return is_causal, best_f, best_p, best_lag
    except Exception:
        return False, 0, 1, 0


def bootstrap_mediation_analysis(x, y, z, max_lag=2, n_bootstrap=1000, confidence_level=0.95):
    """
    Perform bootstrapped mediation analysis to quantify indirect effect.
    """
    try:
        if len(x) < max_lag + 10:
            return None
        
        # Stack arrays and remove NaN
        data = np.column_stack([y, x, z])
        mask = ~np.isnan(data).any(axis=1)
        data = data[mask]
        
        if len(data) < max_lag + 10:
            return None
        
        n = len(data)
        y_vals = data[:, 0]
        x_vals = data[:, 1]
        z_vals = data[:, 2]
        
        # Create lagged variables
        y_current = y_vals[max_lag:]
        
        # Lagged predictors
        x_lagged = []
        z_lagged = []
        y_lagged = []
        
        for lag in range(1, max_lag + 1):
            x_lagged.append(x_vals[max_lag - lag:n - lag])
            z_lagged.append(z_vals[max_lag - lag:n - lag])
            y_lagged.append(y_vals[max_lag - lag:n - lag])
        
        X_x = np.column_stack(x_lagged + y_lagged)
        X_z = np.column_stack(z_lagged + y_lagged)
        X_xz = np.column_stack(x_lagged + z_lagged + y_lagged)
        
        X_x = sm.add_constant(X_x)
        X_z = sm.add_constant(X_z)
        X_xz = sm.add_constant(X_xz)
        
        # Path a: X → M (effect of X on mediator Z)
        model_a = OLS(z_vals[max_lag:], X_x).fit()
        a_coef = np.mean(model_a.params[1:max_lag+1])  # Average effect across lags
        
        # Path b: M → Y (effect of mediator Z on Y, controlling for X)
        model_b = OLS(y_current, X_xz).fit()
        b_coef = np.mean(model_b.params[max_lag+1:2*max_lag+1])  # Z coefficients
        
        # Path c: X → Y (total effect)
        X_total = np.column_stack(x_lagged + y_lagged)
        X_total = sm.add_constant(X_total)
        model_c = OLS(y_current, X_total).fit()
        c_coef = np.mean(model_c.params[1:max_lag+1])  # Total effect
        
        # Path c': X → Y controlling for M (direct effect)
        c_prime_coef = np.mean(model_b.params[1:max_lag+1])  # X coefficients in full model
        
        # Mediation effects
        indirect_effect = a_coef * b_coef  # Indirect effect through mediator
        direct_effect = c_prime_coef       # Direct effect
        total_effect = c_coef              # Total effect
        
        # Bootstrap confidence intervals for indirect effect
        bootstrap_indirect = []
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            indices = np.random.choice(len(y_current), size=len(y_current), replace=True)
            
            try:
                # Resample all data
                y_boot = y_vals[indices]
                x_boot = x_vals[indices]
                z_boot = z_vals[indices]
                
                # Recreate lagged variables for bootstrap sample
                y_curr_boot = y_boot[max_lag:]
                
                x_lag_boot = []
                z_lag_boot = []
                y_lag_boot = []
                
                for lag in range(1, max_lag + 1):
                    x_lag_boot.append(x_boot[max_lag - lag:len(x_boot) - lag])
                    z_lag_boot.append(z_boot[max_lag - lag:len(z_boot) - lag])
                    y_lag_boot.append(y_boot[max_lag - lag:len(y_boot) - lag])
                
                X_x_boot = sm.add_constant(np.column_stack(x_lag_boot + y_lag_boot))
                X_xz_boot = sm.add_constant(np.column_stack(x_lag_boot + z_lag_boot + y_lag_boot))
                
                # Path a: X → M
                model_a_boot = OLS(z_boot[max_lag:], X_x_boot).fit()
                a_boot = np.mean(model_a_boot.params[1:max_lag+1])
                
                # Path b: M → Y|X
                model_b_boot = OLS(y_curr_boot, X_xz_boot).fit()
                b_boot = np.mean(model_b_boot.params[max_lag+1:2*max_lag+1])
                
                # Indirect effect
                bootstrap_indirect.append(a_boot * b_boot)
            except:
                continue
        
        if len(bootstrap_indirect) < n_bootstrap * 0.5:  # Need at least 50% successful bootstraps
            return None
        
        bootstrap_indirect = np.array(bootstrap_indirect)
        
        # Calculate confidence intervals
        alpha = 1 - confidence_level
        ci_lower = np.percentile(bootstrap_indirect, 100 * alpha / 2)
        ci_upper = np.percentile(bootstrap_indirect, 100 * (1 - alpha / 2))
        
        # Mediation is significant if CI doesn't include 0
        is_significant = not (ci_lower <= 0 <= ci_upper)
        
        # Proportion mediated (handle division by zero)
        if abs(total_effect) > 1e-10:
            proportion_mediated = abs(indirect_effect / total_effect)
        else:
            proportion_mediated = 0
        
        return {
            'direct_effect': direct_effect,
            'indirect_effect': indirect_effect,
            'total_effect': total_effect,
            'proportion_mediated': proportion_mediated,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'is_significant': is_significant,
            'a_path': a_coef,
            'b_path': b_coef}
        
    except Exception as e:
        return None


def detect_and_remove_mediated_edges(G, data_transposed, max_lag=2, p_value_threshold=0.05, 
                                     mediation_threshold=0.5, n_bootstrap=1000):
    """
    Detect mediated relationships using bootstrapped mediation analysis.
    Remove edges where >mediation_threshold of effect is mediated.
    """
    edges_to_remove = []
    mediation_info = {}
    
    # Get all edges
    all_edges = list(G.edges())
    total_edges = len(all_edges)
    
    print(f"\nTesting {total_edges} edges for mediation...")
    
    # Pre-extract data arrays
    data_dict = {col: data_transposed[col].values for col in data_transposed.columns}
    
    tested_count = 0
    
    for cause, effect in all_edges:
        tested_count += 1
        
        if tested_count % 50 == 0:
            print(f"  Progress: {tested_count}/{total_edges} edges tested, {len(edges_to_remove)} marked for removal")
        
        # Find potential mediators
        mediators = set(G.successors(cause)) & set(G.predecessors(effect))
        
        if not mediators:
            mediation_info[(cause, effect)] = {
                'is_mediated': False,
                'mediator': None,
                'proportion_mediated': 0,
                'indirect_effect': 0,
                'direct_effect': 0,
                'ci_lower': None,
                'ci_upper': None}
            continue
        
        # Test each potential mediator with bootstrapping
        best_mediator = None
        best_proportion = 0
        best_result = None
        
        for mediator in mediators:
            result = bootstrap_mediation_analysis(
                data_dict[cause],
                data_dict[effect],
                data_dict[mediator],
                max_lag=max_lag,
                n_bootstrap=n_bootstrap,
                confidence_level=0.95)
            
            if result is None:
                continue
            
            # Check if this mediator explains more than previous ones
            if result['is_significant'] and result['proportion_mediated'] > best_proportion:
                best_proportion = result['proportion_mediated']
                best_mediator = mediator
                best_result = result
        
        # Decide whether to remove edge based on mediation threshold
        if best_result and best_result['is_significant'] and best_proportion > mediation_threshold:
            edges_to_remove.append((cause, effect))
            mediation_info[(cause, effect)] = {
                'is_mediated': True,
                'mediator': best_mediator,
                'proportion_mediated': best_proportion,
                'indirect_effect': best_result['indirect_effect'],
                'direct_effect': best_result['direct_effect'],
                'total_effect': best_result['total_effect'],
                'ci_lower': best_result['ci_lower'],
                'ci_upper': best_result['ci_upper'],
                'a_path': best_result['a_path'],
                'b_path': best_result['b_path']}
        else:
            mediation_info[(cause, effect)] = {
                'is_mediated': False,
                'mediator': best_mediator,
                'proportion_mediated': best_proportion if best_result else 0,
                'indirect_effect': best_result['indirect_effect'] if best_result else 0,
                'direct_effect': best_result['direct_effect'] if best_result else 0,
                'ci_lower': best_result['ci_lower'] if best_result else None,
                'ci_upper': best_result['ci_upper'] if best_result else None}
    
    # Remove mediated edges
    print(f"\n Mediation testing complete")
    print(f"  Total edges tested: {total_edges}")
    print(f"  Mediated edges found (>{mediation_threshold*100:.0f}%): {len(edges_to_remove)}")
    print(f"  Direct edges remaining: {total_edges - len(edges_to_remove)}")
    
    if edges_to_remove:
        print(f"\nRemoving {len(edges_to_remove)} mediated edges...")
        print("\nExamples of removed mediated relationships:")
        for i, (cause, effect) in enumerate(edges_to_remove[:10]):
            info = mediation_info[(cause, effect)]
            mediator = info['mediator']
            prop = info['proportion_mediated']
            print(f"  {cause} → {effect}: {prop*100:.1f}% mediated by {mediator}")
            print(f"    Indirect effect: {info['indirect_effect']:.4f} [95% CI: {info['ci_lower']:.4f}, {info['ci_upper']:.4f}]")
        
        if len(edges_to_remove) > 10:
            print(f"  ... and {len(edges_to_remove) - 10} more")
        
        G.remove_edges_from(edges_to_remove)
    
    # Remove isolated nodes
    nodes_with_edges = set()
    for u, v in G.edges():
        nodes_with_edges.add(u)
        nodes_with_edges.add(v)
    
    nodes_to_remove = set(G.nodes()) - nodes_with_edges
    if nodes_to_remove:
        G.remove_nodes_from(nodes_to_remove)
        print(f"  Removed {len(nodes_to_remove)} nodes that became isolated")
    
    print(f"\n✓ Final network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    return G, mediation_info


def CausalEdge(data, id_column='Protein.Names',
                corr_threshold=0.6, f_stat_threshold=10,
                p_value_threshold=0.05, max_lag=2,
                remove_mediated=True, mediation_threshold=0.5,
                n_bootstrap=1000, fdr_method='fdr_bh'):
    """
    CausalEdge causal analysis with FDR correction and bootstrapped mediation.
    """
    # Transpose data
    print(f"\nOriginal shape: {data.shape}")
    data_transposed = data.set_index(id_column).T
    print(f"Transposed shape: {data_transposed.shape}")
    print(f"Time points: {data_transposed.shape[0]}")
    print(f"Biomarkers: {data_transposed.shape[1]}")

    # Handle missing values
    print("\nHandling missing values...")
    missing_before = data_transposed.isnull().sum().sum()

    missing_pct = data_transposed.isnull().mean()
    biomarkers_to_keep = missing_pct[missing_pct < 0.5].index
    data_transposed = data_transposed[biomarkers_to_keep]

    print(f"Removed {len(missing_pct) - len(biomarkers_to_keep)} biomarkers with >50% missing")

    data_transposed = data_transposed.interpolate(method='linear', axis=0, limit_direction='both')

    missing_after = data_transposed.isnull().sum().sum()
    print(f"Missing values: {missing_before} → {missing_after}")

    # Calculate correlation
    print("\nCalculating correlations...")
    data_array = data_transposed.values
    biomarker_list = list(data_transposed.columns)
    n_biomarkers = len(biomarker_list)

    corr_array = np.corrcoef(data_array.T)

    # Build network
    G = nx.DiGraph()
    G.add_nodes_from(biomarker_list)

    # Test causality
    print("\nTesting Granger causality...")
    print(f"Parameters: corr_threshold={corr_threshold}, f_stat_threshold={f_stat_threshold}")
    print(f"           p_value_threshold={p_value_threshold}, max_lag={max_lag}")
    print(f"           FDR correction method: {fdr_method}")

    # Pre-filter pairs by correlation
    print("Pre-filtering candidate pairs by correlation...")
    candidate_pairs = []

    abs_corr = np.abs(corr_array)
    high_corr_mask = abs_corr >= corr_threshold

    for i in range(n_biomarkers):
        high_corr_mask[i, i] = False

    i_indices, j_indices = np.where(high_corr_mask)

    for idx in range(len(i_indices)):
        i, j = i_indices[idx], j_indices[idx]
        candidate_pairs.append((i, j, corr_array[i, j]))

    print(f"Found {len(candidate_pairs)} candidate pairs to test")

    # Test causality and collect p-values for FDR correction
    total_tests = len(candidate_pairs)
    test_results = []  # Store all test results for FDR correction

    data_arrays = {i: data_transposed.iloc[:, i].values for i in range(n_biomarkers)}

    print("\nPerforming Granger causality tests...")
    for test_idx, (i, j, corr_value) in enumerate(candidate_pairs):
        if (test_idx + 1) % 500 == 0:
            print(f"  Progress: {test_idx+1}/{total_tests} tests completed")

        col1 = biomarker_list[i]
        col2 = biomarker_list[j]

        is_causal, f_stat, p_value, lag = check_granger_causality(
            data_arrays[i],
            data_arrays[j],
            max_lag=max_lag,
            f_stat_threshold=f_stat_threshold,
            p_value_threshold=1.0  )

        # Store result regardless of initial significance
        if f_stat > 0:  # Only store if test was successful
            test_results.append({
                'cause': col1,
                'effect': col2,
                'f_stat': f_stat,
                'p_value': p_value,
                'lag': lag,
                'correlation': corr_value
            })

    print(f" Completed {len(test_results)} valid Granger tests")

    # Apply FDR correction
    print(f"\nApplying {fdr_method.upper()} correction for multiple testing...")
    
    if len(test_results) > 0:
        p_values = [r['p_value'] for r in test_results]
        
        # Perform FDR correction
        reject, p_adjusted, _, _ = multipletests(p_values, alpha=p_value_threshold, method=fdr_method)
        
        # Add corrected p-values to results
        for i, result in enumerate(test_results):
            result['p_adjusted'] = p_adjusted[i]
            result['significant'] = reject[i]
        
        # Count significant results
        n_significant = sum(reject)
        n_before_fdr = sum(p < p_value_threshold for p in p_values)
        
        print(f"  Before FDR correction: {n_before_fdr} significant relationships")
        print(f"  After FDR correction: {n_significant} significant relationships")
        print(f"  Removed {n_before_fdr - n_significant} false positives")
        
        # Add only significant edges to graph
        causal_edges = 0
        for result in test_results:
            if result['significant']:
                G.add_edge(
                    result['cause'],
                    result['effect'],
                    weight=abs(result['correlation']),
                    correlation=float(result['correlation']),
                    color='red' if result['correlation'] < 0 else 'blue',
                    f_stat=result['f_stat'],
                    p_value=result['p_value'],
                    p_adjusted=result['p_adjusted'],
                    lag=result['lag'])
                causal_edges += 1
        
        print(f"  Added {causal_edges} edges to network")
    else:
        print("  No valid test results for FDR correction")

    # Remove isolated nodes
    nodes_with_edges = set()
    for u, v in G.edges():
        nodes_with_edges.add(u)
        nodes_with_edges.add(v)

    nodes_to_remove = set(G.nodes()) - nodes_with_edges
    G.remove_nodes_from(nodes_to_remove)
    print(f"  Removed {len(nodes_to_remove)} isolated nodes")

    print(f"\n Initial network (after FDR): {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Mediation analysis
    mediation_info = {}
    if remove_mediated and G.number_of_edges() > 0:
        G, mediation_info = detect_and_remove_mediated_edges(
            G, data_transposed, max_lag=max_lag, 
            p_value_threshold=p_value_threshold,
            mediation_threshold=mediation_threshold,
            n_bootstrap=n_bootstrap
        )

    # Network metrics
    if G.number_of_nodes() > 0 and G.number_of_edges() > 0:
        print("\n" + "="*70)
        print("Network Summary")
        print("="*70)

        out_degree = dict(G.out_degree())
        in_degree = dict(G.in_degree())

        print("\nTop 5 biomarkers by out-degree (causes many others):")
        sorted_out = sorted(out_degree.items(), key=lambda x: x[1], reverse=True)
        for node, degree in sorted_out[:5]:
            if degree > 0:
                print(f"  {node}: {degree} outgoing edges")

        print("\nTop 5 biomarkers by in-degree (influenced by many others):")
        sorted_in = sorted(in_degree.items(), key=lambda x: x[1], reverse=True)
        for node, degree in sorted_in[:5]:
            if degree > 0:
                print(f"  {node}: {degree} incoming edges")

        edges_list = [(u, v, data['f_stat'], data['correlation'], 
                      data['p_value'], data.get('p_adjusted', data['p_value']), data['lag'])
                      for u, v, data in G.edges(data=True)]
        edges_list.sort(key=lambda x: x[2], reverse=True)

        print("\nTop 10 strongest causal relationships:")
        for u, v, f_stat, corr, p_val, p_adj, lag in edges_list[:min(10, len(edges_list))]:
            direction = "→" if corr > 0 else "⊣"
            print(f"  {u} {direction} {v}: F={f_stat:.2f}, p={p_val:.5f}, p_adj={p_adj:.5f}, lag={lag}")

    # Create results DataFrame
    causal_df = None
    if G.number_of_edges() > 0:
        relationships = []
        
        for u, v, data in G.edges(data=True):
            rel_dict = {
                'Cause': u,
                'Effect': v,
                'Correlation': data['correlation'],
                'Direction': 'Positive' if data['correlation'] > 0 else 'Negative',
                'F-statistic': data['f_stat'],
                'p-value': data['p_value'],
                'p-value_adjusted': data.get('p_adjusted', data['p_value']),
                'Optimal Lag': data['lag'],
                'Is_Mediated': False,
                'Mediator': None,
                'Proportion_Mediated': 0,
                'Indirect_Effect': 0,
                'CI_Lower': None,
                'CI_Upper': None
            }
            relationships.append(rel_dict)

        causal_df = pd.DataFrame(relationships)
        causal_df.sort_values('F-statistic', ascending=False, inplace=True)

        print("\n" + "="*50)
        print("Direct Causal Relationships (Top 20)")
        print("="*50)
        print(causal_df[['Cause', 'Effect', 'Direction', 'F-statistic', 
                        'p-value_adjusted', 'Optimal Lag']].head(20).to_string(index=False))

        causal_df.to_csv('causal_relationships_direct.csv', index=False)
        print(f"\n All {len(causal_df)} direct relationships saved to causal_relationships_direct.csv")

    # Create DataFrame for mediated relationships
    if mediation_info:
        mediated_relationships = []
        
        for (cause, effect), info in mediation_info.items():
            if info['is_mediated']:
                med_dict = {
                    'Cause': cause,
                    'Effect': effect,
                    'Is_Mediated': True,
                    'Mediator': info['mediator'],
                    'Proportion_Mediated': info['proportion_mediated'],
                    'Indirect_Effect': info['indirect_effect'],
                    'Direct_Effect': info['direct_effect'],
                    'Total_Effect': info.get('total_effect', 0),
                    'CI_Lower': info['ci_lower'],
                    'CI_Upper': info['ci_upper'],
                    'A_Path_Coefficient': info.get('a_path', 0),
                    'B_Path_Coefficient': info.get('b_path', 0)
                }
                mediated_relationships.append(med_dict)
        
        if mediated_relationships:
            mediated_df = pd.DataFrame(mediated_relationships)
            mediated_df.sort_values('Proportion_Mediated', ascending=False, inplace=True)
            
            print("\n" + "="*70)
            print(f"Mediated Relationships Removed (Top 10)")
            print("="*70)
            print(mediated_df[['Cause', 'Effect', 'Mediator', 'Proportion_Mediated', 
                              'Indirect_Effect']].head(10).to_string(index=False))
            
            mediated_df.to_csv('causal_relationships_mediated_removed.csv', index=False)
            print(f"\n {len(mediated_df)} mediated relationships saved to causal_relationships_mediated_removed.csv")

    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70)
    return G, causal_df, data_transposed, mediation_info, test_results



print("Upload CSV file:")
uploaded = files.upload()

filename = list(uploaded.keys())[0]
print(f"\n Loaded file: {filename}")

data = pd.read_csv(filename)
print(f" Data shape: {data.shape}")
print(f"\nFirst few rows:")
print(data.head())

# Run analysis
G, causal_df, data_transposed, mediation_info, test_results = CausalEdge(
    data=data,
    id_column='Protein.Names',
    corr_threshold=0.75,
    f_stat_threshold=10,
    p_value_threshold=0.05,
    max_lag=2,
    remove_mediated=True,
    mediation_threshold=0.5,  # Remove edges if >50% mediated
    n_bootstrap=1000,         # Number of bootstrap samples
    fdr_method='fdr_bh'       # Benjamini-Hochberg FDR correction
)


if causal_df is not None and len(causal_df) > 0:
    print("\n" + "="*70)
    print("Results Summary")
    print("="*70)

    print(f"\n Total direct relationships found: {len(causal_df)}")
    print(f" Average F-statistic: {causal_df['F-statistic'].mean():.2f}")
    print(f" Median F-statistic: {causal_df['F-statistic'].median():.2f}")

    pos = (causal_df['Direction'] == 'Positive').sum()
    neg = (causal_df['Direction'] == 'Negative').sum()
    total = len(causal_df)
    print(f"\n ositive relationships: {pos} ({100*pos/total:.1f}%)")
    print(f" Negative relationships: {neg} ({100*neg/total:.1f}%)")

    strong = (causal_df['F-statistic'] > 15).sum()
    print(f"\n Very strong relationships (F>15): {strong}")

    print("\n Lag distribution:")
    lag_counts = causal_df['Optimal Lag'].value_counts().sort_index()
    for lag, count in lag_counts.items():
        print(f"    Lag {lag}: {count} relationships")
    
    # FDR statistics
    print("\n Multiple testing correction:")
    print(f"    Mean adjusted p-value: {causal_df['p-value_adjusted'].mean():.5f}")
    print(f"    Max adjusted p-value: {causal_df['p-value_adjusted'].max():.5f}")
else:
    print("\n" + "="*70)
    print(" No Causal Relationships Found")
    print("="*70)
    print("\nTry adjusting parameters:")
    print("  • Lower corr_threshold to 0.5")
    print("  • Lower f_stat_threshold to 5")
    print("  • Raise p_value_threshold to 0.10")

# Downlaod results
print("\n" + "="*70)
print("Download Files")
print("="*70)

if causal_df is not None:
    try:
        files.download('causal_relationships_direct.csv')
        print(" Downloaded: causal_relationships_direct.csv")
    except:
        print("  Could not auto-download causal_relationships_direct.csv")

try:
    files.download('causal_relationships_mediated_removed.csv')
    print(" Downloaded: causal_relationships_mediated_removed.csv")
except:
    print("  No mediated relationships file to download")
