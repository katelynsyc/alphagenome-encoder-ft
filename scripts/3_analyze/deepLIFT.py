
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
from tangermeme.deep_lift_shap import deep_lift_shap
from tangermeme.ersatz import dinucleotide_shuffle



#import and make the model
model = Beluga()

X = random_one_hot((1, 4, 2000)).type(torch.float32)
X = substitute(X, "GTGACTCATC")

X_attr = deep_lift_shap(model, X, target=267, device='cpu', random_state=0, print_convergence_deltas=True)
X_attr.shape

#dinucleotide_shuffle for reference sequence to preserve GC content, use large enough number of reference sequences (this is diff than shuffling many times)?
seq = one_hot_encode('CATCGACAGACTACGCTAC').unsqueeze(0)
shuf = dinucleotide_shuffle(seq, random_state=0) #you can also add start and end to just shuffle portions of the sequence


#i don't think i need them to do predictions?

#how to predic multi-outputs

from matplotlib import pyplot as plt
import seaborn; seaborn.set_style('whitegrid')
from tangermeme.plot import plot_logo

plt.figure(figsize=(10, 2))
ax = plt.subplot(111)
plot_logo(X_attr[0, :, 950:1050], ax=ax)

plt.xlabel("Genomic Coordinate")
plt.ylabel("Attributions")
plt.title("DeepLIFT Attributions for GM12878 JunD")
plt.ylim(-0.05, 0.35)
plt.show()

X_attr = deep_lift_shap(model, X, target=214, device='cpu',random_state=0) #plotting a diff target

plt.figure(figsize=(10, 2))
ax = plt.subplot(111)
plot_logo(X_attr[0, :, 950:1050], ax=ax)

plt.xlabel("Genomic Coordinate")
plt.ylabel("Attributions")
plt.title("DeepLIFT Attributions for GM12878 ETS1")
plt.ylim(-0.05, 0.35)
plt.show()
#MAKE SURE YOU COMPARED ACROSS SAME REFERENCE SEQUENCES --use random_state to an integer and use the the same number of shuffles

#tangermeme itself has marginalization, ablation, spacing, saturation mutagenesis 
#for design: it has a greedy search algorithm to design sequence that gives desired predictions