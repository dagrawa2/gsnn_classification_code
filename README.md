# A Classification of G-invariant Shallow Neural Networks
Devanshu Agrawal & James Ostrowski

### Abstract

When trying to fit a deep neural network (DNN) to a G-invariant target function with respect to a group G, it only makes sense to constrain the DNN to be G-invariant as well. 
However, there can be many different ways to do this, thus raising the problem of "G-invariant neural architecture design": 
What is the optimal G-invariant architecture for a given problem? 
Before we can consider the optimization problem itself, we must understand the search space, the architectures in it, and how they relate to one another. 
In this paper, we take a first step towards this goal; 
we prove a theorem that gives a classification of all G-invariant single-hidden-layer or "shallow" neural network (G-SNN) architectures with ReLU activation for any finite orthogonal group G. 
The proof is based on a correspondence of every G-SNN to a signed permutation representation of G acting on the hidden neurons. 
The classification is equivalently given in terms of the first cohomology classes of G, thus admitting a topological interpretation. 
Based on a code implementation, we enumerate the G-SNN architectures for some example groups G and visualize their structure. 
We draw the network morphisms between the enumerated architectures that can be leveraged during neural architecture search (NAS). 
Finally, we prove that architectures corresponding to inequivalent cohomology classes in a given cohomology ring coincide in function space only when their weight matrices are zero, and we discuss the implications of this in the context of NAS.

### Description

This repository includes code and scripts to generate the figures in a paper.


### Requirements

- [GAP >= 4.11.0](https://www.gap-system.org/Releases/4.11.0.html)
- python >= 3.8.10
- matplotlib
- networkx
- numpy
- pandas
- scipy


# Generating figures

To generate the irreducible G-SNN architectures for all groups G listed in Table 1 of the paper, run the following:

    bash run.sh

This will create and populate the `results` directory. 
To generate the constrained weight patterns and cohomology class illustrations (including Figures 1 and 4-6 of the paper) for all groups G in the results directory, run the following:

    python plot.py

This will create and populate the `plots` directory. 
The constrained weight patterns and cohomology class illustrations for group `G` will then be found in the `plots/vis/G/weights` and `plots/vis/G/cohomology` subdirectories respectively. 
To generate the contour plots in Figures 2 and 8 of the paper, run the following:

    python plot_contour.py

These plots will be found in the `plots/vis/C_6/contour` and `plots/vis/D_6/contour` subdirectories for Figures 1 and 8 respectively. 
Finally, to generate the network morphism graphs in Figures 3 and 7, run the following:

    python plot_morphism.py

The figures will be found in the `plots/vis/C_6` and `plots/vis/D_6` subdirectories. 
Note that the scripts `plot_contour.py` and `plot_morphism.py` generate no results but only plot them; 
the results for these plots were calculated manually using the theorems in the associated paper.
