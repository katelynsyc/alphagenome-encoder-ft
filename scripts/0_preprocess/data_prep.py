#data prep to split into the testing sets

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from scipy.stats import stats

def excel_to_tsv(mpra_activity_file, sequences_file):
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

    merged = merged.rename(columns={'RNA/DNA ratio': 'Leaf', 'Unnamed: 2': 'MG', 'Unnamed: 3': 'Br','Unnamed: 4': 'RR', 'Unique barcodes recovered from RNA-seq libraries': 'Unique Barcodes'})
    raw = merged.copy()
    
    activity_cols = ['Leaf', 'MG', 'Br', 'RR'] #these are RNA/DNA, we want to make them log2(RNA/DNA)

    log2_transformed = merged[activity_cols].copy().apply(pd.to_numeric, errors='coerce')
    #convert to log2 values with psuedocounts
    merged[activity_cols] = np.log2(log2_transformed.to_numpy(dtype=float) + 0.1) #log2((RNA/DNA) + 0.1)
    merged = merged.reset_index(drop=True)

    merged.to_csv('/home/kachu/alphagenome-encoder-ft/metadata/all_log2_activity.tsv', sep='\t', index=False) #save as tsv
    
    normalized = merged
    plot_raw_vs_normalized_expression(raw, normalized)
    return merged

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

        m, b, r_value, _, _ = stats.linregress(raw_i, normalized_i)
        x_line = np.array([raw_i.min(), raw_i.max()])
        ax.plot(x_line, m * x_line + b, color="black", linewidth=1)

        ax.set_title(f"{tissue}  (r={r_value:.3f})")
        ax.set_xlabel("Raw Gene Expression (RNA/DNA)")
        ax.set_ylabel("Normalized log2(RNA/DNA)")
        ax.annotate(f"r² = {r_value**2:.3f}", xy=(0.05, 0.92), xycoords="axes fraction", fontsize=9)
        ax.annotate(f"y = {m:.2f}x + {b:.2f}", xy=(0.05, 0.85), xycoords="axes fraction", fontsize=9)

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
    print(fragmentActivity.head())

    zero_leaf_act =  fragmentActivity[fragmentActivity[tissue_type] == 0] #filtered df with just
    print(zero_leaf_act)
    leaf_zeroes = set(zero_leaf_act['Fragment'].to_list()) #compare the ID to those in the deng data to see if they kept them
    zero_leaf_above_10 = fragmentActivity[(fragmentActivity[tissue_type] == 0) & (fragmentActivity['Unique Barcodes'] >= 10)] 
    print(f">=10 and 0 {tissue_type}: {zero_leaf_above_10['Fragment'].to_list()}")
    
    #for their train file
    deng_train = pd.read_csv(deng_train, sep='\t')
    print(deng_train.head())
    their_train = set(deng_train['Name'].to_list())

    leaf_zero_deng_train = leaf_zeroes & their_train
    print("Intersection of 0 {tissue_type} Activity and Deng Train")
    print(leaf_zero_deng_train) #these say leaf but func can be used for any of the 4 conditions
    print(len(leaf_zero_deng_train))

    #for test file
    deng_test = pd.read_csv(deng_test, sep='\t')
    print(deng_test.head())
    their_test = set(deng_test['ID'].to_list())

    leaf_zero_deng_test = leaf_zeroes & their_test
    print("Intersection of 0 {tissue_type} Activity and Deng Test")
    print(leaf_zero_deng_test)
    print(len(leaf_zero_deng_test))



    
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

def plot__overall_distribution(data, barcode_thresh): #plots one combined dataset of the gene expresion
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
                
    plt.title(f'Overall Distribution (Barcode Threshold: {barcode_thresh})', fontsize=12, fontweight='bold')
    plt.xlabel('Expression (log2 (RNA/DNA + 0.1))', fontsize=10)
    plt.ylabel('Density', fontsize=10)
    plt.legend(loc='upper right', fontsize=8)
        
    # add median lines
    for condition, color in zip(conditions, colors):
        values = data[condition].dropna()
        if len(values) > 0:
            median_val = values.median()
            ax.axvline(median_val, color=color, linestyle='--', 
                        alpha=0.7, linewidth=1)
            
    plt.savefig(f'results/plots/allchromosomes{barcode_thresh}thresh.png', dpi=300) 
    plt.close(fig)
   


def plot_chrom_distributions(chromosomes, barcode_thresh): #plots distributions of each chromosome to visualize if there is enough dynamic range
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
        ax.set_xlabel('Expression log2(RNA/DNA + 0.1)', fontsize=10)
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
    
    plt.suptitle(f'Gene Expression Distribution by Chromosome and Condition (Barcode Threshold: {barcode_thresh})', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
          
    plt.savefig(f'results/plots/indiv_chromosomes{barcode_thresh}thresh.png', dpi=300)   

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
    metadata_path = "/home/kachu/alphagenome-encoder-ft/metadata"
    
    mpra_activity_file = metadata_path + "/Supplementary Full Dataset 2.xlsx"
    sequences_file = metadata_path + "/Supplementary Data Set 1.xlsx"
    deng_train = metadata_path + "/train.txt"
    log_2_activity = metadata_path + "/all_log2_activity.tsv"
    deng_test = metadata_path + "/test.txt"
    untransformed_diffs = pd.DataFrame(leaf_activity_diff(deng_test, log_2_activity)).transpose()
    print(f"Max diff: {untransformed_diffs['diff'].max()}")

    with pd.option_context("display.precision", 15):
        print(untransformed_diffs)

    #compare_zero_processing(mpra_activity_file, deng_train, deng_test, 'Br')

    all_data = excel_to_tsv(mpra_activity_file, sequences_file)

    # barcode_threshold = 10
    # above_ten_thresh = filter_threshold(all_data, barcode_threshold) #start with >= 10 unique barcodes
    # chrom_dict, val_chrom, test_chrom, chrom_percentages = split_chroms(above_ten_thresh) #this is the strict >=10 set

    # save_splits(above_ten_thresh, metadata_path + "/10_barcode_thresh")
    # write_chrom_percentages(chrom_dict, chrom_percentages, barcode_threshold, metadata_path + "/chromosome_readout_percentages")
    # #print(chrom_dict.keys())

    # plot__overall_distribution(above_ten_thresh, barcode_threshold) #plot all the data
    # plot_chrom_distributions(chrom_dict, barcode_threshold)

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


                              
