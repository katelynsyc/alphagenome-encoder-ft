#data prep to split into the testing sets

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from scipy import stats
import json

def excel_to_tsv(mpra_activity_file, sequences_file, pseudocount):
    fragmentActivity = pd.read_excel(mpra_activity_file, skiprows=1)
    allFragmentSeqs = pd.read_excel(sequences_file, skiprows=1)
    
    # merge on fragment name, keep only those with recorded fragment activity
    merged = pd.merge(
        fragmentActivity, 
        allFragmentSeqs[['Name', 'Chr', 'Sequence']], 
        left_on='Fragment',
        right_on='Name',
        how='left'
    )
    
    merged = merged.drop(columns='Name') #double cols of the names
    merged = merged.drop(merged.index[0]) #bc excel sheet had strange formatting with double label rows
    #remove the 35S enhancer row because that was a control and not tissue, nor developmental stage specific
    merged = merged[merged['Fragment'] != '35S enhancer']
    merged = merged.reset_index(drop=True) #so raw and normalized share the same index after dropped rows above

    merged = merged.rename(columns={'RNA/DNA ratio': 'Leaf', 'Unnamed: 2': 'MG', 'Unnamed: 3': 'Br','Unnamed: 4': 'RR', 'Unique barcodes recovered from RNA-seq libraries': 'Unique Barcodes'})
    raw = merged.copy()
    #print(raw.head())
    raw_frags = set(raw['Fragment'].to_list())
    
    activity_cols = ['Leaf', 'MG', 'Br', 'RR'] #these are RNA/DNA, we want to make them log2(RNA/DNA)

    if not pseudocount: #if we want to use theirs and will later do imputation, there will be a lot of NA values for those
        nan_log2 = merged[activity_cols].replace(0, np.nan).apply(pd.to_numeric, errors='coerce') # Replaces all 0s with NaN, scoped to the activity columns only
        merged[activity_cols] = np.log2(nan_log2.to_numpy(dtype=float)) #log2(RNA/DNA), 0s stay NaN for later imputation
        merged = merged.reset_index(drop=True)

        #merged.to_csv('/home/kachu/alphagenome-encoder-ft/metadata/no_pseudo_log2_activity.tsv', sep='\t', index=False) #save as tsv
        return merged

    else:
        log2_transformed = merged[activity_cols].copy().apply(pd.to_numeric, errors='coerce')
        #convert to log2 values with psuedocounts
        merged[activity_cols] = np.log2(log2_transformed.to_numpy(dtype=float) + 0.1) #log2((RNA/DNA) + 0.1)
        merged = merged.reset_index(drop=True)

        merged.to_csv('/home/kachu/alphagenome-encoder-ft/metadata/all_log2_activity.tsv', sep='\t', index=False) #save as tsv
        
        normalized = merged
        #print(normalized.head())
        norm_frags = set(normalized['Fragment'].to_list())
        is_lined_up = len(raw_frags) == len(norm_frags) and all(a == b for a, b in zip(raw_frags, norm_frags))
        #print(f"Exact same order: {is_lined_up}")
        #plot_raw_vs_normalized_expression(raw, normalized)
        return merged

def acr_excel_to_tsv(excel_file, output_path, sheet_name='ACR sequence library'):
    """Extract the ACR sequence library sheet (media-3.xlsx) into a tsv with
    columns: id, orientation, light, dark, warm, cold, maize, chromosome,
    start, end, GC content (%), sequence. The sheet has a title/description
    in the first rows and a merged 'log2(enhancer strength)' group header,
    so the real column names live on the row right above the data.
    """
    df = pd.read_excel(excel_file, sheet_name=sheet_name, header=4)
    cols = ['id', 'orientation', 'light', 'dark', 'warm', 'cold', 'maize',
            'chromosome', 'GC content (%)', 'sequence'] #might need GC content?
    df = df[cols]
    df.to_csv(output_path, sep='\t', index=False)
    print(f"Wrote {df.shape[0]} rows to {output_path}")
    return df

def count_na_per_condition(excel_file, sheet_name='ACR sequence library'):
    """Print the number (and %) of N/A log2(enhancer strength) values per
    condition (light, dark, warm, cold, maize) in the ACR sequence library sheet.
    """
    df = pd.read_excel(excel_file, sheet_name=sheet_name, header=4)
    conditions = ['light', 'dark', 'warm', 'cold', 'maize']
    na_counts = {}
    print(f"Total rows: {len(df)}")
    for cond in conditions:
        na_count = df[cond].isna().sum()
        na_counts[cond] = na_count
        print(f"{cond}: {na_count} NA values ({na_count / len(df) * 100:.2f}%)")
    return na_counts

def count_rows_by_na_amount(excel_file, sheet_name='ACR sequence library'):
    """Print how many rows have exactly 0, 1, 2, 3, 4, or 5 N/A condition
    values (light, dark, warm, cold, maize) and how many have at least that
    many, in the ACR sequence library sheet.
    """
    df = pd.read_excel(excel_file, sheet_name=sheet_name, header=4)
    conditions = ['light', 'dark', 'warm', 'cold', 'maize']
    na_per_row = df[conditions].isna().sum(axis=1)

    exact_counts = {}
    print(f"Total rows: {len(df)}")
    for n in range(len(conditions) + 1):
        count = int((na_per_row == n).sum())
        exact_counts[n] = count
        print(f"{n} NAs: {count} rows ({count / len(df) * 100:.2f}%)")

    at_least_counts = {}
    for n in range(1, len(conditions) + 1):
        count = int((na_per_row >= n).sum())
        at_least_counts[n] = count
        print(f">= {n} NAs: {count} rows ({count / len(df) * 100:.2f}%)")

    return exact_counts, at_least_counts

def load_pseudocount_log2(log2_excel, sequences_file, readcount_file, output_path=None):
    """Attach Chr, Sequence, Unique Barcodes to a pseudocount log2 activity table
    (Fragment, Leaf, Fruit, MG, Br, RR) by looking them up per-Fragment.

    Chr/Sequence come from sequences_file (Supplementary Data Set 1.xlsx, Name/Chr/Sequence
    columns). Unique Barcodes isn't in that file -- it comes from readcount_file
    (Supplementary Dataset 2-ReadCount-RPM-ratio-log2ratio.xlsx), the same active-only
    fragment set log2_excel was itself computed from.
    """
    pseudo = pd.read_excel(log2_excel)
    pseudo = pseudo[pseudo['Fragment'] != '35S enhancer'].reset_index(drop=True)  # control, not a real fragment

    seqs = pd.read_excel(sequences_file, skiprows=1)[['Name', 'Chr', 'Sequence']]
    counts = pd.read_excel(readcount_file, header=1)[
        ['Fragment', 'Unique barcodes recovered from RNA-seq libraries']
    ].rename(columns={'Unique barcodes recovered from RNA-seq libraries': 'Unique Barcodes'})

    merged = pd.merge(pseudo, seqs, left_on='Fragment', right_on='Name', how='left').drop(columns='Name')
    merged = pd.merge(merged, counts, on='Fragment', how='left')

    missing = merged.loc[merged['Chr'].isna() | merged['Unique Barcodes'].isna(), 'Fragment'].tolist()
    if missing:
        print(f"Warning: {len(missing)} fragments missing Chr/Sequence or Unique Barcodes: {missing[:5]}")

    if output_path:
        merged.to_csv(output_path, sep='\t', index=False)
        print(f"Wrote {merged.shape[0]} rows to {output_path}")

    return merged


def compute_pseudocount_log2_from_readcounts(readcount_file, sequences_file, output_path=None, pseudocount=1):
    """Recompute per-fragment log2(RNA/DNA) activity directly from the raw
    'RNA DNA ReadCount All.xlsx' sheet, pooling replicates the correct way instead of
    using the workbook's own precomputed RPM/ratio/log2 columns.

    The sheet has merged header cells (e.g. 'DNA-seq (ReadCount)', 'RNA-seq (RPM)') sitting
    above per-library sub-columns (DNA-1, Leaf-1, ...), so it's read with a 2-row header --
    that keeps every (group, library) column name unique even though 'DNA-1' etc. repeat
    under both the ReadCount and RPM groups.

    Each step below is printed (labeled) as it's computed:
      1. total counts  -- recover each library's true sequencing depth as
                          (count / RPM) * 1e6. This is constant down the column (RPM was
                          computed from it), so any row would do; we take the median across
                          all nonzero rows for robustness. One total per library: 3 for DNA,
                          12 for RNA (Leaf/MG/Br/RR x 3 reps).
      2. summed totals -- pool replicates: sum DNA's 3 library totals into one shared
                          denominator, and separately sum each state's 3 library totals.
                          Also pool the raw counts the same way, per fragment. Fruit pools
                          all 9 MG+Br+RR replicate libraries together the same way (not an
                          average of the three states' own log2 values -- see write-up above).
      3. norms         -- (pooled count + pseudocount) / pooled total x 1e6, per fragment,
                          done once for DNA, once per state, and once for pooled Fruit.
      4. log2 activity -- log2(RNA_norm[state] / DNA_norm), per fragment, per state/Fruit.

    Chr/Sequence are then attached by matching Fragment against sequences_file's Name column
    (Supplementary Data Set 1.xlsx), the same lookup load_pseudocount_log2 uses.

    Returns a DataFrame with columns Fragment, Leaf, MG, Br, RR, Fruit, Unique Barcodes, Chr, Sequence.
    """
    data = pd.read_excel(readcount_file, header=[0, 1])
    fragment_col = ('Unnamed: 0_level_0', 'Fragment')
    data = data[data[fragment_col] != '35S enhancer'].reset_index(drop=True)  # control row, not a real fragment

    states = ['Leaf', 'MG', 'Br', 'RR']
    replicates = [1, 2, 3]

    # ---- Step 1: recover each library's true total (sequencing depth) ----
    def library_total(count_group, rpm_group, library):
        counts = data[(count_group, library)]
        rpms = data[(rpm_group, library)]
        nonzero = rpms > 0  # a fragment absent from this library has count=rpm=0; skip it
        totals = counts[nonzero] / rpms[nonzero] * 1e6
        return totals.median()

    dna_totals = {rep: library_total('DNA-seq (ReadCount)', 'DNA-seq (RPM)', f'DNA-{rep}') for rep in replicates}
    print("Step 1 - recovered DNA library totals:")
    for rep, total in dna_totals.items():
        print(f"  DNA-{rep}: {total:,.1f}")

    rna_totals = {
        state: {rep: library_total('RNA-seq (ReadCount)', 'RNA-seq (RPM)', f'{state}-{rep}') for rep in replicates}
        for state in states
    }
    print("Step 1 - recovered RNA library totals:")
    for state in states:
        for rep, total in rna_totals[state].items():
            print(f"  {state}-{rep}: {total:,.1f}")

    # ---- Step 2: pool replicates -- sum raw counts and sum library totals ----
    dna_count_sum = sum(data[('DNA-seq (ReadCount)', f'DNA-{rep}')] for rep in replicates)  # per-fragment
    dna_total_sum = sum(dna_totals.values())  # one number, shared across every state

    print(f"\nStep 2 - pooled DNA total (sum of 3 reps): {dna_total_sum:,.1f}")
    print("Step 2 - pooled DNA counts (per fragment), preview:")
    print(dna_count_sum.head())

    rna_count_sum = {}
    rna_total_sum = {}
    for state in states:
        rna_count_sum[state] = sum(data[('RNA-seq (ReadCount)', f'{state}-{rep}')] for rep in replicates)
        rna_total_sum[state] = sum(rna_totals[state].values())
        print(f"\nStep 2 - pooled {state} total (sum of 3 reps): {rna_total_sum[state]:,.1f}")
        print(f"Step 2 - pooled {state} counts (per fragment), preview:")
        print(rna_count_sum[state].head())

    fruit_states = ['MG', 'Br', 'RR']
    fruit_total_sum = sum(rna_total_sum[state] for state in fruit_states)  # all 9 MG+Br+RR libraries, pooled
    fruit_count_sum = sum(rna_count_sum[state] for state in fruit_states)  # per-fragment
    print(f"\nStep 2 - pooled Fruit total (sum of MG+Br+RR's 9 reps): {fruit_total_sum:,.1f}")
    print("Step 2 - pooled Fruit counts (per fragment), preview:")
    print(fruit_count_sum.head())

    # ---- Step 3: normalize each pooled count by its pooled total, RPM-style ----
    dna_norm = (dna_count_sum + pseudocount) / dna_total_sum * 1e6
    print("\nStep 3 - DNA_norm (per fragment), preview:")
    print(dna_norm.head())

    rna_norm = {}
    for state in states:
        rna_norm[state] = (rna_count_sum[state] + pseudocount) / rna_total_sum[state] * 1e6
        print(f"Step 3 - {state}_norm (per fragment), preview:")
        print(rna_norm[state].head())

    fruit_norm = (fruit_count_sum + pseudocount) / fruit_total_sum * 1e6
    print("Step 3 - Fruit_norm (per fragment), preview:")
    print(fruit_norm.head())

    # ---- Step 4: log2(RNA_norm / DNA_norm) activity, per fragment, per state ----
    result = pd.DataFrame({'Fragment': data[fragment_col]})
    for state in states:
        result[state] = np.log2(rna_norm[state] / dna_norm)
        print(f"\nStep 4 - {state} log2 activity, preview:")
        print(result[state].head())

    result['Fruit'] = np.log2(fruit_norm / dna_norm)
    print("\nStep 4 - Fruit log2 activity, preview:")
    print(result['Fruit'].head())

    result['Unique Barcodes'] = data[('RNA-seq (ReadCount)', 'Unique barcodes recovered from RNA-seq libraries')]

    # ---- attach Chr/Sequence by matching Fragment against sequences_file's Name column ----
    seqs = pd.read_excel(sequences_file, skiprows=1)[['Name', 'Chr', 'Sequence']]
    result = pd.merge(result, seqs, left_on='Fragment', right_on='Name', how='left').drop(columns='Name')

    missing = result.loc[result['Chr'].isna(), 'Fragment'].tolist()
    if missing:
        print(f"Warning: {len(missing)} fragments missing Chr/Sequence: {missing[:5]}")

    result = result[['Fragment', 'Leaf', 'MG', 'Br', 'RR', 'Fruit', 'Unique Barcodes', 'Chr', 'Sequence']]

    if output_path:
        result.to_csv(output_path, sep='\t', index=False)
        print(f"\nWrote {result.shape[0]} rows to {output_path}")

    return result


def compute_replicate_log2_activity(readcount_file, pseudocount=1, include_pooled_fruit=True):
    """Compute per-replicate log2(RNA_norm/DNA_norm) activity, one column per (state, replicate)
    e.g. 'Leaf-1', 'Leaf-2', 'Leaf-3', 'MG-1', ... -- using a shared pooled DNA normalization
    (summed across the 3 DNA replicates) as the denominator for every RNA replicate. This mirrors
    compute_pseudocount_log2_from_readcounts's DNA-side normalization, just without summing the
    RNA replicates together first, so each replicate keeps its own activity value.

    If include_pooled_fruit is True, also adds 'Fruit-1', 'Fruit-2', 'Fruit-3' columns: for each
    replicate index, MG/Br/RR's counts and totals at that same replicate index are pooled together
    (mirroring how compute_pseudocount_log2_from_readcounts pools all 9 MG+Br+RR libraries into a
    single 'Fruit' condition), giving 3 pooled-fruit-stage replicate values that can be correlated
    against each other to get a reliability ceiling for the pooled Fruit condition itself.
    """
    data = pd.read_excel(readcount_file, header=[0, 1])
    fragment_col = ('Unnamed: 0_level_0', 'Fragment')
    data = data[data[fragment_col] != '35S enhancer'].reset_index(drop=True)  # control row, not a real fragment

    states = ['Leaf', 'MG', 'Br', 'RR']
    fruit_states = ['MG', 'Br', 'RR']
    replicates = [1, 2, 3]

    def library_total(count_group, rpm_group, library):
        counts = data[(count_group, library)]
        rpms = data[(rpm_group, library)]
        nonzero = rpms > 0  # a fragment absent from this library has count=rpm=0; skip it
        totals = counts[nonzero] / rpms[nonzero] * 1e6
        return totals.median()

    dna_totals = {rep: library_total('DNA-seq (ReadCount)', 'DNA-seq (RPM)', f'DNA-{rep}') for rep in replicates}
    dna_count_sum = sum(data[('DNA-seq (ReadCount)', f'DNA-{rep}')] for rep in replicates)
    dna_total_sum = sum(dna_totals.values())
    dna_norm = (dna_count_sum + pseudocount) / dna_total_sum * 1e6  # shared across every state/replicate

    result = pd.DataFrame({'Fragment': data[fragment_col]})
    rna_counts = {}  # (state, rep) -> raw counts, kept around to pool into Fruit-{rep} below
    rna_totals = {}  # (state, rep) -> library total
    for state in states:
        for rep in replicates:
            library = f'{state}-{rep}'
            total = library_total('RNA-seq (ReadCount)', 'RNA-seq (RPM)', library)
            counts = data[('RNA-seq (ReadCount)', library)]
            rna_counts[(state, rep)] = counts
            rna_totals[(state, rep)] = total
            rna_norm = (counts + pseudocount) / total * 1e6
            result[library] = np.log2(rna_norm / dna_norm)

    if include_pooled_fruit:
        for rep in replicates:
            pooled_count = sum(rna_counts[(state, rep)] for state in fruit_states)
            pooled_total = sum(rna_totals[(state, rep)] for state in fruit_states)
            pooled_norm = (pooled_count + pseudocount) / pooled_total * 1e6
            result[f'Fruit-{rep}'] = np.log2(pooled_norm / dna_norm)

    result['Unique Barcodes'] = data[('RNA-seq (ReadCount)', 'Unique barcodes recovered from RNA-seq libraries')]

    return result


def _spearman_brown_ceiling(pairwise_rs, n_replicates):
    """Collapse several pairwise single-replicate correlations into the reliability
    ceiling for the n_replicates-pooled measurement.

    The Spearman-Brown prophecy formula (n*r / (1 + (n-1)*r)) takes a single
    single-replicate reliability r, so the pairwise r's are first collapsed into one
    representative r via a Fisher z-transform average (arctanh -> mean -> tanh), which
    is the statistically correct way to average correlations -- the plain arithmetic
    mean of r's is slightly biased low.
    """
    z_mean = np.mean(np.arctanh(pairwise_rs))
    r_mean = np.tanh(z_mean)
    return (n_replicates * r_mean) / (1 + (n_replicates - 1) * r_mean)


def compute_replicate_correlations(readcount_file, pseudocount=1, method='pearson', min_barcodes=None, spearman_brown=False, include_pooled_fruit=False):
    """For each triplicate condition (Leaf, MG, Br, RR), compute the pairwise correlation
    between every pair of replicates' log2(RNA/DNA) activity values (1v2, 1v3, 2v3), then
    average those three pairwise correlations into a single per-condition replicate
    correlation -- one number each for Leaf, MG, Br, and RR.

    method is passed straight to pd.Series.corr ('pearson', 'spearman', or 'kendall').
    min_barcodes, if given, restricts to fragments with Unique Barcodes >= min_barcodes
    before correlating (same threshold semantics as filter_threshold).
    spearman_brown, if True, also reports the Spearman-Brown corrected reliability ceiling
    for the 3-replicate-pooled measurement (see _spearman_brown_ceiling) -- this is the
    correct noise ceiling to compare a model's pearson against, since the model is scored
    against the pooled/mean-of-3-replicates target, not against a single noisy replicate.
    include_pooled_fruit, if True, adds a 'Fruit' condition built from MG/Br/RR pooled
    together within each replicate index (see compute_replicate_log2_activity), so its
    reliability ceiling can be compared directly against the individual MG/Br/RR ceilings --
    this is the ceiling relevant to a model trained against the single pooled Fruit target
    rather than the three separate developmental stages.

    Returns {state: {'pairwise': {'1v2': r, '1v3': r, '2v3': r}, 'mean': avg_r
                      [, 'spearman_brown': ceiling]}}.
    """
    replicate_activity = compute_replicate_log2_activity(readcount_file, pseudocount=pseudocount, include_pooled_fruit=include_pooled_fruit)
    if min_barcodes is not None:
        replicate_activity = filter_threshold(replicate_activity, min_barcodes)

    states = ['Leaf', 'MG', 'Br', 'RR'] + (['Fruit'] if include_pooled_fruit else [])
    n_replicates = 3
    pairs = [(1, 2), (1, 3), (2, 3)]

    correlations = {}
    for state in states:
        pairwise = {
            f'{rep_a}v{rep_b}': replicate_activity[f'{state}-{rep_a}'].corr(
                replicate_activity[f'{state}-{rep_b}'], method=method
            )
            for rep_a, rep_b in pairs
        }
        correlations[state] = {
            'pairwise': pairwise,
            'mean': float(np.mean(list(pairwise.values()))),
        }
        message = f"{state}: mean replicate correlation = {correlations[state]['mean']:.4f} (pairwise: {pairwise})"

        if spearman_brown:
            ceiling = _spearman_brown_ceiling(list(pairwise.values()), n_replicates=n_replicates)
            correlations[state]['spearman_brown'] = float(ceiling)
            message += f" | Spearman-Brown ceiling (n={n_replicates}) = {ceiling:.4f}"

        print(message)

    return correlations


def plot_replicate_ceiling_bar(readcount_file, states, thresholds=(5, 10), pseudocount=1, method='pearson',
                                title=None, output_path=None):
    """Grouped bar chart of the Spearman-Brown-corrected replicate correlation ceiling
    (compute_replicate_correlations(..., spearman_brown=True)) for each condition in
    states, with one bar per barcode threshold in thresholds side by side per condition.

    Pass states=['Leaf', 'MG', 'Br', 'RR'] for the per-developmental-stage ceilings, or
    states=['Leaf', 'Fruit'] for the pooled-Fruit-condition ceiling instead (Fruit is
    automatically included/pooled via compute_replicate_log2_activity's include_pooled_fruit
    when 'Fruit' is present in states).
    """
    threshold_colors = ['forestgreen', 'lightgreen']
    include_pooled_fruit = 'Fruit' in states

    ceilings = {}
    for threshold in thresholds:
        correlations = compute_replicate_correlations(
            readcount_file, pseudocount=pseudocount, method=method,
            min_barcodes=threshold, spearman_brown=True, include_pooled_fruit=include_pooled_fruit,
        )
        ceilings[threshold] = [correlations[state]['spearman_brown'] for state in states]

    x = np.arange(len(states))
    n_thresholds = len(thresholds)
    width = 0.8 / n_thresholds

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, threshold in enumerate(thresholds):
        offset = (i - (n_thresholds - 1) / 2) * width
        bars = ax.bar(
            x + offset, ceilings[threshold], width,
            color=threshold_colors[i % len(threshold_colors)],
            label=f'>= {threshold} barcodes',
        )
        for bar, value in zip(bars, ceilings[threshold]):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f'{value:.2f}',
                    ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(states)
    ax.set_ylabel('Spearman-Brown corrected replicate correlation ceiling')
    ax.set_title(title or f"Replicate Correlation Ceiling by Barcode Threshold ({', '.join(states)})")
    ax.set_ylim(0, max(v for values in ceilings.values() for v in values) * 1.15)  # headroom for value labels
    ax.legend(loc='upper left', fontsize=8)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300)
    return fig


def plot_raw_vs_normalized_expression(raw, normalized):
    tissues = ["Leaf", "MG", "Br", "RR"]
    colors = ["#2ecc71", "#e74c3c", "#e67e22", "#e74c3c"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for i, (tissue, color, ax) in enumerate(zip(tissues, colors, axes)):
        raw_i = pd.to_numeric(raw[tissue], errors="coerce")
        normalized_i = pd.to_numeric(normalized[tissue], errors="coerce")
        mask = raw_i.notna() & normalized_i.notna()
        raw_i = raw_i[mask].to_numpy(dtype=float)
        normalized_i = normalized_i[mask].to_numpy(dtype=float)
        ax.scatter(raw_i, normalized_i, color=color, s=8, alpha=0.4)

        # m, b, r_value, _, _ = stats.linregress(raw_i, normalized_i)
        # x_line = np.array([raw_i.min(), raw_i.max()])
        # ax.plot(x_line, m * x_line + b, color="black", linewidth=1)

        ax.set_title(f"{tissue}")
        ax.set_xlabel("Raw Gene Expression (RNA/DNA)")
        ax.set_ylabel("Normalized log2(RNA/DNA)")
        # ax.annotate(f"r² = {r_value**2:.3f}", xy=(0.05, 0.92), xycoords="axes fraction", fontsize=9)
        # ax.annotate(f"y = {m:.2f}x + {b:.2f}", xy=(0.05, 0.85), xycoords="axes fraction", fontsize=9)

    fig.suptitle(f"Raw vs. Normalized Gene Expression", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"results/plots/raw_vs_normalized_expression.png", dpi=300)
    plt.close(fig)


def compare_zero_processing(mpra_activity_file, deng_train, deng_test, tissue_type):
    fragmentActivity = pd.read_excel(mpra_activity_file, skiprows=1)
    fragmentActivity = fragmentActivity.drop(fragmentActivity.index[0]) #bc excel sheet had strange formatting with double label rows
    #remove the 35S enhancer row because that was a control and not tissue, nor developmental stage specific
    fragmentActivity = fragmentActivity[fragmentActivity['Fragment'] != '35S enhancer']
    
    fragmentActivity = fragmentActivity.rename(columns={'RNA/DNA ratio': 'Leaf', 'Unnamed: 2': 'MG', 'Unnamed: 3': 'Br','Unnamed: 4': 'RR', 'Unique barcodes recovered from RNA-seq libraries': 'Unique Barcodes'})
    #print(fragmentActivity.head())
    
    if tissue_type == "Fruit": #sum the fruit activity raw values together
        fragmentActivity ['Fruit'] = fragmentActivity[['MG', 'Br', 'RR']].sum(axis=1)
        
    zero_tissue =  fragmentActivity[fragmentActivity[tissue_type] == 0] #filtered df with just
    print(zero_tissue)
    print(f"Rows with 0 Fruit Activity {zero_tissue.shape[0]}")
    zero_tissue = set(zero_tissue['Fragment'].to_list()) #compare the ID to those in the deng data to see if they kept them
    zero_above_10 = fragmentActivity[(fragmentActivity[tissue_type] == 0) & (fragmentActivity['Unique Barcodes'] >= 10)] 
    print(f">=10 and 0 {tissue_type}: {zero_above_10['Fragment'].to_list()}")
    print(f">=10 and 0 {tissue_type} Number: {len(zero_above_10['Fragment'].to_list())}")
    
    #for their train file
    deng_train = pd.read_csv(deng_train, sep='\t')
    #print(deng_train.head())
    their_train = set(deng_train['Name'].to_list())

    zero_deng_train = zero_tissue & their_train
    print("Intersection of 0 {tissue_type} Activity and Deng Train")
    print(zero_deng_train) #these say leaf but func can be used for any of the 4 conditions
    print(len(zero_deng_train))
    zero_kept_train = deng_train[deng_train['Name'].isin(zero_deng_train)]
    print("Kept Train Sequences and their Values")
    print(zero_kept_train)

    #for test file
    deng_test = pd.read_csv(deng_test, sep='\t')
    #print(deng_test.head())
    their_test = set(deng_test['ID'].to_list())

    zero_deng_test = zero_tissue & their_test
    print("Intersection of 0 {tissue_type} Activity and Deng Test")
    print(zero_deng_test)
    print(len(zero_deng_test))
    zero_kept_test = deng_test[deng_test['ID'].isin(zero_deng_test)]
    print("Kept Test Sequences and their Values")
    print(zero_kept_test)

    return zero_kept_train, zero_kept_test

def write_imputation_dict(mpra_activity_file, deng_train, deng_test, output_path):
    """Build {fragment_name: {'Leaf': Leaf_activity, 'Fruit': Fruit_activity}} for every
    fragment that had a raw 0 activity value (Leaf or Fruit) but was still kept -- with a
    Deng et al. imputed non-zero activity value -- in their train/test files, and write it
    to a JSON file at output_path.
    """
    imputation_dict: dict[str, dict[str, float]] = {}

    for tissue_type in ("Leaf", "Fruit"):
        zero_kept_train, zero_kept_test = compare_zero_processing(
            mpra_activity_file, deng_train, deng_test, tissue_type
        )
        for _, row in zero_kept_train.iterrows():
            imputation_dict[row["Name"]] = {
                "Leaf": float(row["Leaf_activity"]),
                "Fruit": float(row["Fruit_activity"]),
            }
        for _, row in zero_kept_test.iterrows():
            imputation_dict[row["ID"]] = {
                "Leaf": float(row["Leaf_activity"]),
                "Fruit": float(row["Fruit_activity"]),
            }

    with open(output_path, "w") as f:
        json.dump(imputation_dict, f, indent=2)

    print(f"Wrote imputation dict with {len(imputation_dict)} entries to {output_path}")
    return imputation_dict

def write_imputed_activity_tsv(input_tsv, imputation_dict, output_path):
    """Read input_tsv (e.g. all_log2_activity.tsv, left untouched), add a
    Fruit = mean(MG, Br, RR) column, overwrite Leaf/Fruit with Deng et al.'s imputed
    values for fragments in imputation_dict, and save the result to output_path.
    mydata.py can then read Leaf/Fruit directly with no runtime imputation needed.
    """
    #data = pd.read_csv(input_tsv, sep='\t')
    data = input_tsv
    print(f"Pure Log2 Before imputing: {input_tsv}")
    data['Fruit'] = data[['MG', 'Br', 'RR']].mean(axis=1)
    columns_titles = ["Fragment", "Leaf", "MG", "Br", "RR", "Fruit", "Unique Barcodes",	"Chr", "Sequence"] #put the fruit values nearby
    data=data.reindex(columns=columns_titles)

    for fragment, activity in imputation_dict.items():
        mask = data['Fragment'] == fragment
        data.loc[mask, 'Leaf'] = activity['Leaf']
        data.loc[mask, 'Fruit'] = activity['Fruit']

    data.to_csv(output_path, sep='\t', index=False)
    print(f"Wrote imputed activity tsv ({data.shape[0]} rows) to {output_path}")
    return data

def filter_threshold(data, barcode_threshold): #returns a dataframe that filtered based on this # of barcodes
    above_thresh = data[data['Unique Barcodes'] >= barcode_threshold].copy()
    # print(f"\nFiltered data shape: {above_thresh.shape}")
    # print(above_thresh.head())
    return above_thresh

def split_chroms(data) -> tuple[dict, str, str]: #for 80-10-10 split of train, val, test
    #will return a tuple with dictionary of the chromosomes, then str key names to the chromosomes of the val and test sets respectively
    total_seq_num = len(data)
    target_percent = 10.0

    chromosomes = {chromosome: fragment for chromosome, fragment in  data.groupby('Chr')} #make dict of dataframes based on the chromosome
    #calculate percentages for each chromosome
    chrom_percentages = {}

    # print("Number of Sequences:")
    for chrom in chromosomes.keys(): #print chromosome and associated # of seqs
        percent = len(chromosomes[chrom]) * 100 / total_seq_num
        chrom_percentages[chrom] = percent
        #print(f"{chrom}: {len(chromosomes[chrom])} sequences, or {percent:.2f}%")
    
    sorted_chroms = sorted(chrom_percentages.items(), #sort by distance from 10%
                          key=lambda x: abs(x[1] - target_percent))
    
    test_chrom = sorted_chroms[0][0] #this is the name in str of the chromosome picked
    val_chrom = sorted_chroms[1][0]

    # print(f"--- Selected Chromosomes ---")
    # print(f"Test set: Chr {test_chrom} ({chrom_percentages[test_chrom]:.2f}%)")
    # print(f"Validation set: Chr {val_chrom} ({chrom_percentages[val_chrom]:.2f}%)")
    
    return (chromosomes, val_chrom, test_chrom, chrom_percentages)  #dictionary with dataframes of each chromosome

def make_splits(chromosomes, val_chrom, test_chrom):
    val_data = chromosomes[val_chrom]
    test_data = chromosomes[test_chrom]
    train_chroms = [chrom for chrom in chromosomes.keys() if chrom not in [test_chrom, val_chrom]] #all the other chromosomes
    train_data = pd.concat([chromosomes[c] for c in train_chroms])
    splits = {
        'train': train_data,
        'val': val_data,
        'test': test_data,
        'chromosomes': chromosomes
    }

    return splits

def save_splits(data, output_path):
    # cols = ['Fragment',	'Leaf', 'MG', 'Br',	'RR', 'Unique Barcodes', 'Chr', 'Sequence']
    # cleaned_df = data[cols].rename(columns={'Unique barcodes recovered from RNA-seq libraries': 'Unique Barcodes'})
    # cleaned_df.to_csv(output_path, sep='\t', index=False)
    data.to_csv(output_path, sep='\t', index=False)

def write_chrom_percentages(chromosomes, chrom_percentages, barcode_threshold, output_path):
    total_seqs = sum(len(df) for df in chromosomes.values())
    lines = [
        "Chromosome Split Summary",
        f"Total sequences: {total_seqs}",
        f"Number of Unique Barcodes Threshold: {barcode_threshold}"
        "",
        f"{'Chromosome':<15} {'Count':>8} {'Percent':>10}", #table headers
    ]
    for chrom in sorted(chromosomes.keys()):
        count = len(chromosomes[chrom]) #length of 
        pct = chrom_percentages[chrom]
        lines.append(f"{chrom:<15} {count:>8} {pct:>9.2f}%")

    with open(output_path, 'w') as f:
        f.write("\n".join(lines) + "\n")

def plot__overall_distribution(data, barcode_thresh=None): #plots one combined dataset of the gene expresion
    fig, ax = plt.subplots(figsize=(10, 6))
    # define conditions and colors
    conditions = ['Leaf', 'MG', 'Br', 'RR']
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12']  # green, red, blue, orange

    for condition, color in zip(conditions, colors):
            # get non-zero, non-NaN values
            values = data[condition].dropna()
            
            if len(values) > 0:
                # plot density/histogram
                values.plot.kde(ax=ax, label=condition, color=color, 
                               linewidth=2.5, alpha=0.8)

    if barcode_thresh is not None:
        plt.title(f'Overall Distribution (Barcode Threshold: {barcode_thresh})', fontsize=12, fontweight='bold')
    else:
        plt.title(f'Overall Distribution)', fontsize=12, fontweight='bold')
   
    plt.xlabel('Expression (log2 (RNA norm/DNA norm))', fontsize=10)
    plt.ylabel('Density', fontsize=10)
    plt.legend(loc='upper right', fontsize=8)
        
    # add median lines
    for condition, color in zip(conditions, colors):
        values = data[condition].dropna()
        if len(values) > 0:
            median_val = values.median()
            ax.axvline(median_val, color=color, linestyle='--', 
                        alpha=0.7, linewidth=1)
            
    #plt.savefig(f'results/plots/allchromosomes{barcode_thresh}thresh.png', dpi=300) 
    #plt.close(fig)
   


def plot_chrom_distributions(chromosomes, barcode_thresh=None): #plots distributions of each chromosome to visualize if there is enough dynamic range
    sns.set_style("whitegrid")
    
    # define conditions and colors
    conditions = ['Leaf', 'MG', 'Br', 'RR']
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12']  # green, red, blue, orange
    
    # create subplots: 4 rows x 3 columns for 12 chromosomes
    fig, axes = plt.subplots(4, 3, figsize=(18, 16))
    axes = axes.flatten()

    for ind, chrom in enumerate(chromosomes):
        chrom_data = chromosomes[chrom]
        ax = axes[ind]

        for condition, color in zip(conditions, colors):
            # get non-zero, non-NaN values
            values = chrom_data[condition].dropna()
            
            if len(values) > 0:
                # plot the kde line
                values.plot.kde(ax=ax, label=condition, color=color, 
                               linewidth=2.5, alpha=0.8)
                
        ax.set_title(f'{chrom} (n={len(chrom_data)})', fontsize=12, fontweight='bold')
        ax.set_xlabel('Expression (log2 (RNA norm/DNA norm), Pseudocount = 1)', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # add median lines
        for condition, color in zip(conditions, colors):
            values = chrom_data[condition].dropna()
            if len(values) > 0:
                median_val = values.median()
                ax.axvline(median_val, color=color, linestyle='--', 
                          alpha=0.7, linewidth=1)
    if barcode_thresh is not None: 
        plt.suptitle(f'Gene Expression Distribution by Chromosome and Condition (Barcode Threshold: {barcode_thresh})', 
                 fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(f'results/plots/indiv_chromosomes{barcode_thresh}thresh.png', dpi=300) 
    else:
        plt.suptitle(f'Gene Expression Distribution by Chromosome and Condition)', 
                 fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(f'results/plots/indiv_chromosomes.png', dpi=300) 
    
          
    #plt.savefig(f'results/plots/indiv_chromosomes{barcode_thresh}thresh.png', dpi=300)   

def plot_acr_overall_distribution(data, title=None, output_path=None): #plots one combined dataset of the ACR log2(enhancer strength) values, mirrors plot__overall_distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    conditions = ['light', 'dark', 'warm', 'cold', 'maize']
    colors = ['#2a78d6', '#1baf7a', '#eda100', '#008300', '#4a3aa7']  # blue, aqua, yellow, green, violet

    for condition, color in zip(conditions, colors):
        values = data[condition].dropna()
        if len(values) > 0:
            values.plot.kde(ax=ax, label=condition, color=color,
                           linewidth=2.5, alpha=0.8)

    plt.title(title or 'ACR Sequence Library: Overall Distribution', fontsize=12, fontweight='bold')
    plt.xlabel('log2(enhancer strength)', fontsize=10)
    plt.ylabel('Density', fontsize=10)
    plt.legend(loc='upper right', fontsize=8)

    # add median lines
    for condition, color in zip(conditions, colors):
        values = data[condition].dropna()
        if len(values) > 0:
            median_val = values.median()
            ax.axvline(median_val, color=color, linestyle='--',
                        alpha=0.7, linewidth=1)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300)
    return fig


def plot_acr_distribution_by_split(data, output_path=None): #plots modelling_data_tamsACR.tsv's log2 enrichment: one overall panel plus one panel per train/val/test 'set' value, each with all five conditions overlaid
    sns.set_style("whitegrid")

    conditions = ['enrichment_light', 'enrichment_dark', 'enrichment_warm', 'enrichment_cold', 'enrichment_maize']
    labels = ['light', 'dark', 'warm', 'cold', 'maize']
    colors = ['#2a78d6', '#1baf7a', '#eda100', '#008300', '#4a3aa7']  # blue, aqua, yellow, green, violet

    panels = [('Overall', data)] + [
        (split_name.capitalize(), data[data['set'] == split_name]) for split_name in ('train', 'val', 'test')
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, (panel_name, panel_data) in zip(axes, panels):
        for condition, label, color in zip(conditions, labels, colors):
            values = panel_data[condition].dropna()
            if len(values) > 0:
                values.plot.kde(ax=ax, label=label, color=color,
                               linewidth=2.5, alpha=0.8)

        ax.set_title(f'{panel_name} (n={len(panel_data)})', fontsize=12, fontweight='bold')
        ax.set_xlabel('log2 enrichment', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

        # add median lines
        for condition, label, color in zip(conditions, labels, colors):
            values = panel_data[condition].dropna()
            if len(values) > 0:
                median_val = values.median()
                ax.axvline(median_val, color=color, linestyle='--',
                          alpha=0.7, linewidth=1)

    plt.suptitle('ACR Modelling Data: log2 Enrichment Distribution by Condition and Split',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300)
    return fig

def find_condition_diff(data, condition_1, condition_2, output_path=None):
    df = pd.read_csv(data, sep='\t')
    cond1 = df[condition_1]
    cond2 = df[condition_2]
    
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(8, 8))
    plt.scatter(cond1, cond2, alpha=0.5, s=10)

    # Add diagonal line (y=x) - where warm == cold
    min_val = min(cond1.min(), cond2.min())
    max_val = max(cond1.max(), cond2.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='No difference')

    plt.xlabel(f'{condition_1} (log2)')
    plt.ylabel(f'{condition_2} (log2)')
    plt.title('Condition Difference Impact on Expression')
    plt.legend()
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=300)

    largest_diff_ind = (abs(df[condition_1] - df[condition_2])).idxmax()
    diff = df[condition_1][largest_diff_ind] - df[condition_2][largest_diff_ind]
    seq_id = df['id'][largest_diff_ind]
    print(f"Largest difference for {condition_1} and {condition_2} is {seq_id} at {diff}")

def select_chromosomes(chromosome_list, all_data): #takes in tsv of all_data and then just returns data associated with chromosome list
    filtered_df = all_data[all_data['Chr'].isin(chromosome_list)]
    return filtered_df


def leaf_activity_diff(
    test_tsv: str,
    all_activity_tsv: str,
    test_pseudocount: float = 0.0,
    all_pseudocount: float = 0.1,
) -> dict[str, dict[str, float]]:
    """Undo log2 and compute per-fragment leaf activity difference.

    Inverse is 2^x - pseudocount.  all_log2_activity.tsv was built with
    log2(RNA/DNA + 0.1), so all_pseudocount=0.1 by default.  If test.txt
    used no pseudocount, leave test_pseudocount=0.0.

    Returns a dict keyed by fragment ID present in both files:
        {id: {"test_leaf": float, "all_leaf": float, "diff": float}}
    where diff = test_leaf - all_leaf (both on linear RNA/DNA scale).
    """
    test_df = pd.read_csv(test_tsv, sep="\t")
    test_leaf = {
        row["ID"]: 2.0 ** row["Leaf_activity"] - test_pseudocount
        for _, row in test_df.iterrows()
    }

    all_df = pd.read_csv(all_activity_tsv, sep="\t")
    all_leaf = {
        row["Fragment"]: 2.0 ** row["Leaf"] - all_pseudocount
        for _, row in all_df.iterrows()
    }

    common = test_leaf.keys() & all_leaf.keys()
    return {
        frag: {
            "test_leaf": test_leaf[frag],
            "all_leaf":  all_leaf[frag],
            "diff":      test_leaf[frag] - all_leaf[frag],
        }
        for frag in sorted(common)
    }


def main():
    #metadata_path = "/home/kachu/alphagenome-encoder-ft/metadata"
    metadata_path = "/grid/koo/home/kachu/projects/alphagenome-encoder-ft/metadata"

    
    mpra_activity_file = metadata_path + "/Supplementary Full Dataset 2.xlsx"
    sequences_file = metadata_path + "/Supplementary Data Set 1.xlsx"
    deng_train = metadata_path + "/train.txt"
    log_2_activity = metadata_path + "/all_log2_activity.tsv"
    deng_test = metadata_path + "/test.txt"
    active_readcount_file = metadata_path + "/Supplementary Dataset 2-ReadCount-RPM-ratio-log2ratio.xlsx"
    log2_pseudocounted_file = metadata_path + "/log2_pseudocounted.xlsx"
    full_readcount_file = metadata_path + "/RNA DNA ReadCount All.xlsx"

    
    #differences in leaf activity
    # untransformed_diffs = pd.DataFrame(leaf_activity_diff(deng_test, log_2_activity)).transpose()
    # print(f"Max diff: {untransformed_diffs['diff'].max()}")

    # with pd.option_context("display.precision", 15):
    #     print(untransformed_diffs)

    #load_pseudocount_log2(log2_pseudocounted_file, sequences_file, active_readcount_file, metadata_path + "/pseudocounted_log2.tsv")
    all_data = compute_pseudocount_log2_from_readcounts(full_readcount_file, sequences_file, metadata_path + "/full_pseudocount_log2.tsv", pseudocount=1)

    # all_data = excel_to_tsv(mpra_activity_file, sequences_file, False) #true that you want the std log2 and not the pseudocount
    # imputation_dict = write_imputation_dict(mpra_activity_file, deng_train, deng_test, metadata_path + "/imputation_dict.json") #this stays the same
    # write_imputed_activity_tsv(all_data, imputation_dict, metadata_path + "/all_log2_activity_imputed.tsv")

    #compare_zero_processing(mpra_activity_file, deng_train, deng_test, 'Leaf')
    

    barcode_threshold = 10
    above_ten_thresh = filter_threshold(all_data, barcode_threshold) #start with >= 10 unique barcodes
    chrom_dict, val_chrom, test_chrom, chrom_percentages = split_chroms(above_ten_thresh) #this is the strict >=10 set

    # save_splits(above_ten_thresh, metadata_path + "/10_barcode_thresh")
    # write_chrom_percentages(chrom_dict, chrom_percentages, barcode_threshold, metadata_path + "/chromosome_readout_percentages")
    # #print(chrom_dict.keys())

    plot__overall_distribution(above_ten_thresh, barcode_threshold) #plot all the data
    plot_chrom_distributions(chrom_dict, barcode_threshold)

    # ACR sequence library: overall light/dark/warm/cold/maize distribution
    acr_data = pd.read_csv(metadata_path + "/acr_sequence_library.tsv", sep='\t')
    plot_acr_overall_distribution(acr_data, output_path="results/plots/acr_overall_distribution.png")

    # ACR modelling data: overall + train/val/test split distributions (5 conditions each)
    acr_split_data = pd.read_csv(metadata_path + "/modelling_data_tamsACR.tsv", sep='\t')
    plot_acr_distribution_by_split(acr_split_data, output_path="results/plots/acr_distribution_by_split.png")

    # barcode_threshold = 5

    # training_chroms = [str(key) for key in chrom_dict]
    # training_chroms.remove(val_chrom)
    # training_chroms.remove(test_chrom)

    # #only take those from the training chromosomes
    # train_chroms = select_chromosomes(training_chroms, all_data)
    # five_thresh_train = filter_threshold(train_chroms, barcode_threshold) #start with >= 5 unique barcodes from the training set
    # #merge this with the strict >= 10 barcodes for all
    # above_five = pd.concat([above_ten_thresh, five_thresh_train]).drop_duplicates().reset_index(drop=True) #combine them
    # above_five = above_five.sort_values(by='Chr') #group by chromosome together, sort alphabetically
    # save_splits(above_five, metadata_path + "/5_barcode_thresh")

    # relaxed_training, val_chrom, test_chrom, chrom_percentages = split_chroms(above_five) #this is the relaxed >=5 set
    # #print(relaxed_training)

    # #then plot
    # plot__overall_distribution(above_five, barcode_threshold) #plot all the data
    # plot_chrom_distributions(relaxed_training, barcode_threshold) #there should only be 10 of the 12 axes filled out here because just the training chromosomes

    # #we want to count how many sequences were added from the previous 
    # print(f'Strict >= 10 unique barcodes: {above_ten_thresh.shape[0]} sequences')
    # print(f'Relaxed to >= 5 unique barcodes: {above_five.shape[0]} sequences')



    

    
    


if __name__ == "__main__":
    main()


                              
