# [Paper: Attention Output Sparse Autoencoders](https://arxiv.org/abs/2406.17759)

Welcome from ICML 2024! In this repository, you will find the code to
reproduce key results accompanying "Interpreting Attention Layer Outputs
with Sparse Autoencoders" by Kissane et al. 2024. 

* [Paper](https://arxiv.org/abs/2406.17759)
* [Blog posts](https://www.alignmentforum.org/s/FzGeLpkzDgzGhigLm)
* [Attention Circuit Explorer](https://robertzk.github.io/circuit-explorer) ([Loom video example](https://www.loom.com/share/9836bdc0398b4178bc7ae91f0ab244a9))

## Contents

* `train.py` contains code to train GPT-2 Small Attention Output SAEs (Table 2).
* `visualize_gpt2.py` contains code to generate [feature dashboards](https://ckkissane.github.io/attn-sae-gpt2-small-viz/) for GPT-2 Small. It builds on top of Callum McDougall's early [`sae_visualizer`](https://github.com/callummcdougall/sae_visualizer), but our dashboards are customized for Attention Output SAEs, and support visualization of the DFA by source position.
* `induction_feature_deep_dive.py` contains code for the induction feature deep dive (Section 3.3), including specificity and sensitivity analysis (Figure 2). 
* `top_feature_analysis.py` contains code to generate dashboards for the top 10 features attributed to every head in GPT-2 Small (Section 4.1).
* `long_prefix_induction.py` contains code to reproduce all of the results confirming that head 5.1 in GPT-2 Small specializes in long prefix induction, while 5.5 primarily does short prefix induction (Figure 4, Figure 19).
* `ioi_circuit_analysis.py` contains code for the key IOI circuit analysis results (Section 4.3). This includes identifying causally relevant SAE features via zero ablations (Figure 16a) and confirming that the IOI circuit's positional signal depends on the " and" token (Figure 5). We leverage `ioi_dataset.py`, from Callum McDougall's [ARENA](https://github.com/callummcdougall/ARENA_3.0/blob/main/chapter1_transformer_interp/exercises/part3_indirect_object_identification/ioi_dataset.py), to generate the IOI dataset.

