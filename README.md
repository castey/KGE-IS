# Comparing KG Embeddings

This folder contains code to measure embedding drift between triples, using TransE with a fixed seed for reproducibility, on the FB15k-237, FB15k-238, and FB15k-239 datasets.

Developed using python version 3.11.11

## Training and Embeddings

### save_drift_data.py
Calls generate_drift_data.py and saves the result to a json/csv file named drift_data.json/csv. The resulting file size is quite large, over 4GB (json) or 1.5GB (csv).

Usage: `python save_drift_data.py [--format <json|csv>] [--filename <output file>] [--epochs <int>]`

### generate_drift_data.py
Contains functions to train on each dataset and returns an object containing each triple which contain embeddings from each dataset `{ triple->dataset->head[], relation[], tail[] }`

The embeddings present in the final result represent triples from 237, which trivialy exist in the intersection of all three datasets: T<sub>n</sub> ∈ 237 ∩ 238 ∩ 239, due to the nested subset relationship 237 ⊆ 238 ⊆ 239.

e.g.
```
{
    "triple_string": {
        "dataset_237": {"head": embedding[], "relation": embedding[], "tail": embedding[]},
        "dataset_238": {"head": embedding[], "relation": embedding[], "tail": embedding[]},
        "dataset_239": {"head": embedding[], "relation": embedding[], "tail": embedding[]},
    },
    ...
}
```

Note: the script is pointed to the `../dataset/` directory.

## Deterministic Training

### grab_matrices.py
Runs pipeline() on the 237 dataset with num_epochs=0. It then grabs the model and saves to a file. It then does the same thing except with num_epochs set to default. To make pykeen deterministic set random.seed() and torch.manual_seed() to any number.

### compare_matrices.py
This script compares the files outputted by grab_matrices.py and shows which are identical and which are different. The resultant parameters are always the same for any given number of training epochs demonstrating deterministic behavior.

## Calculation and Visualization

### run_drift_analysis.sh
Pipeline generating data and plots given some number of training epochs.

Usage: `./run_drift_analysis <n>`

### calc_drift.py 
Computes the Average Paired Distance Function between embedding vectors for aligned triples shared across multiple knowledge graph datasets.

Outputs results as a markdown file containing a table with the mean and standard deviation and a distribution plot.

<!--<img src = "./img/LatexADF.png" width = "380"></img>-->

Usage: `python calc_drift.py [--filepath <path>] [--spo <head|relation|tail>] [--filename <output.png>]`
<br>

### Justifying Number of Training Epochs

To illustrate the effect of training duration on embedding drift, the following plots were generated using 5 and 100 training epochs. The head and tail distances for 5 epochs appear to show bimodal distributions whereas for 100 epochs the distributions resemble a Gaussian distribution. Additionally training runs were done for 1, 5, 10, 25, 50, 100, and 200 epochs and the mean/sd plotted.


### Mean and Standard Deviations for 5 and 100 epochs.

<!-- Heads -->
<table>
  <tr>
    <th colspan="2" style="text-align:left;">Heads</th>
  </tr>
  <tr>
    <td align="center" width="50%" valign="top">
      <strong>5 Epochs</strong><br>
      <table align="center">
        <tr><th>Dataset Pair</th><th>μ (Mean)</th><th>σ (Std Dev)</th></tr>
        <tr><td><strong>Δ₁</strong></td><td>0.680030</td><td>0.313747</td></tr>
        <tr><td><strong>Δ₂</strong></td><td>0.682184</td><td>0.313614</td></tr>
        <tr><td><strong>Δ₃</strong></td><td>0.439329</td><td>0.064016</td></tr>
      </table>
      <br>
      <img src="./img/drift_data_5epochs_head.png" width="100%">
    </td>
    <td align="center" width="50%" valign="top">
      <strong>100 Epochs</strong><br>
      <table align="center">
        <tr><th>Dataset Pair</th><th>μ (Mean)</th><th>σ (Std Dev)</th></tr>
        <tr><td><strong>Δ₁</strong></td><td>0.899643</td><td>0.131742</td></tr>
        <tr><td><strong>Δ₂</strong></td><td>0.899607</td><td>0.129738</td></tr>
        <tr><td><strong>Δ₃</strong></td><td>0.814815</td><td>0.130666</td></tr>
      </table>
      <br>
      <img src="./img/drift_histogram_heads_100epochs.png" width="100%">
    </td>
  </tr>
</table>

---

<!-- Tails -->
<table>
  <tr>
    <th colspan="2" style="text-align:left;">Tails</th>
  </tr>
  <tr>
    <td align="center" width="50%" valign="top">
      <strong>5 Epochs</strong><br>
      <table align="center">
        <tr><th>Dataset Pair</th><th>μ (Mean)</th><th>σ (Std Dev)</th></tr>
        <tr><td><strong>Δ₁</strong></td><td>0.643354</td><td>0.307334</td></tr>
        <tr><td><strong>Δ₂</strong></td><td>0.646124</td><td>0.309677</td></tr>
        <tr><td><strong>Δ₃</strong></td><td>0.411066</td><td>0.082295</td></tr>
      </table>
      <br>
      <img src="./img/drift_data_5epochs_tail.png" width="100%">
    </td>
    <td align="center" width="50%" valign="top">
      <strong>100 Epochs</strong><br>
      <table align="center">
        <tr><th>Dataset Pair</th><th>μ (Mean)</th><th>σ (Std Dev)</th></tr>
        <tr><td><strong>Δ₁</strong></td><td>0.863759</td><td>0.146335</td></tr>
        <tr><td><strong>Δ₂</strong></td><td>0.865926</td><td>0.146631</td></tr>
        <tr><td><strong>Δ₃</strong></td><td>0.769000</td><td>0.147069</td></tr>
      </table>
      <br>
      <img src="./img/drift_histogram_tails_100epochs.png" width="100%">
    </td>
  </tr>
</table>

---

<!-- Relations -->
<table>
  <tr>
    <th colspan="2" style="text-align:left;">Relations</th>
  </tr>
  <tr>
    <td align="center" width="50%" valign="top">
      <strong>5 Epochs</strong><br>
      <table align="center">
        <tr><th>Dataset Pair</th><th>μ (Mean)</th><th>σ (Std Dev)</th></tr>
        <tr><td><strong>Δ₁</strong></td><td>0.501356</td><td>0.161970</td></tr>
        <tr><td><strong>Δ₂</strong></td><td>0.497328</td><td>0.168527</td></tr>
        <tr><td><strong>Δ₃</strong></td><td>0.376040</td><td>0.096940</td></tr>
      </table>
      <br>
      <img src="./img/drift_data_5epochs_relation.png" width="100%">
    </td>
    <td align="center" width="50%" valign="top">
      <strong>100 Epochs</strong><br>
      <table align="center">
        <tr><th>Dataset Pair</th><th>μ (Mean)</th><th>σ (Std Dev)</th></tr>
        <tr><td><strong>Δ₁</strong></td><td>0.865690</td><td>0.402051</td></tr>
        <tr><td><strong>Δ₂</strong></td><td>0.861736</td><td>0.395032</td></tr>
        <tr><td><strong>Δ₃</strong></td><td>0.759481</td><td>0.326970</td></tr>
      </table>
      <br>
      <img src="./img/drift_histogram_relations_100epochs.png" width="100%">
    </td>
  </tr>
</table>

---

### Mean and Standard Deviation across Epochs

The following plots show the drift and standard deviation generated by a model trained using increasing `num_epoch`.  
Data and plots for each run can be found in the `img/` directory.

<!-- Heads Mean vs Std -->
<table>
  <tr>
    <th colspan="2" style="text-align:left;">Heads: Mean & Std Dev</th>
  </tr>
  <tr>
    <td align="center" width="50%">
      <strong>Mean Drift</strong><br>
      <img src="./img/head_mean_drift.png" width="100%">
    </td>
    <td align="center" width="50%">
      <strong>Standard Deviation</strong><br>
      <img src="./img/head_std_drift.png" width="100%">
    </td>
  </tr>
</table>


---

<!-- Tails Mean vs Std -->
<table>
  <tr>
    <th colspan="2" style="text-align:left;">Tails: Mean & Std Dev</th>
  </tr>
  <tr>
    <td align="center" width="50%">
      <strong>Mean Drift</strong><br>
      <img src="./img/tail_mean_drift.png" width="100%">
    </td>
    <td align="center" width="50%">
      <strong>Standard Deviation</strong><br>
      <img src="./img/tail_std_drift.png" width="100%">
    </td>
  </tr>
</table>

---

<!-- Relations Mean vs Std -->
<table>
  <tr>
    <th colspan="2" style="text-align:left;">Relations: Mean & Std Dev</th>
  </tr>
  <tr>
    <td align="center" width="50%">
      <strong>Mean Drift</strong><br>
      <img src="./img/relation_mean_drift.png" width="100%">
    </td>
    <td align="center" width="50%">
      <strong>Standard Deviation</strong><br>
      <img src="./img/relation_std_drift.png" width="100%">
    </td>
  </tr>
</table>

<!-- Heads and Tails top drifters -->
### Top 10 Drifted Heads and Tails  
Bar graphs showing the top and bottom 10 drifted heads and tails for each comparison.  
100 Epochs used in the training step.

<!-- Top Drift -->
<table>
  <tr>
    <th colspan="2" style="text-align:left;">Top 10 Highest Drift: FB15k-237 vs FB15k-238</th>
  </tr>
  <tr>
    <td align="center" width="50%">
      <strong>Heads</strong><br>
      <img src="./output/head_wikidata_237_vs_238_highest_top.png" width="100%">
    </td>
    <td align="center" width="50%">
      <strong>Tails</strong><br>
      <img src="./output/tail_wikidata_237_vs_238_highest_top.png" width="100%">
    </td>
  </tr>
</table>

<br>

<table>
  <tr>
    <th colspan="2" style="text-align:left;">Top 10 Highest Drift: FB15k-237 vs FB15k-239</th>
  </tr>
  <tr>
    <td align="center" width="50%">
      <strong>Heads</strong><br>
      <img src="./output/head_wikidata_237_vs_239_highest_top.png" width="100%">
    </td>
    <td align="center" width="50%">
      <strong>Tails</strong><br>
      <img src="./output/tail_wikidata_237_vs_239_highest_top.png" width="100%">
    </td>
  </tr>
</table>

<br>

<table>
  <tr>
    <th colspan="2" style="text-align:left;">Top 10 Highest Drift: FB15k-238 vs FB15k-239</th>
  </tr>
  <tr>
    <td align="center" width="50%">
      <strong>Heads</strong><br>
      <img src="./output/head_wikidata_238_vs_239_highest_top.png" width="100%">
    </td>
    <td align="center" width="50%">
      <strong>Tails</strong><br>
      <img src="./output/tail_wikidata_238_vs_239_highest_top.png" width="100%">
    </td>
  </tr>
</table>

<!-- Lowest Drift -->
<br><br>
### Bottom 10 Drifted Heads and Tails

<table>
  <tr>
    <th colspan="2" style="text-align:left;">Bottom 10 Drift: FB15k-237 vs FB15k-238</th>
  </tr>
  <tr>
    <td align="center" width="50%">
      <strong>Heads</strong><br>
      <img src="./output/head_wikidata_237_vs_238_lowest_bottom.png" width="100%">
    </td>
    <td align="center" width="50%">
      <strong>Tails</strong><br>
      <img src="./output/tail_wikidata_237_vs_238_lowest_bottom.png" width="100%">
    </td>
  </tr>
</table>

<br>

<table>
  <tr>
    <th colspan="2" style="text-align:left;">Bottom 10 Drift: FB15k-237 vs FB15k-239</th>
  </tr>
  <tr>
    <td align="center" width="50%">
      <strong>Heads</strong><br>
      <img src="./output/head_wikidata_237_vs_239_lowest_bottom.png" width="100%">
    </td>
    <td align="center" width="50%">
      <strong>Tails</strong><br>
      <img src="./output/tail_wikidata_237_vs_239_lowest_bottom.png" width="100%">
    </td>
  </tr>
</table>

<br>

<table>
  <tr>
    <th colspan="2" style="text-align:left;">Bottom 10 Drift: FB15k-238 vs FB15k-239</th>
  </tr>
  <tr>
    <td align="center" width="50%">
      <strong>Heads</strong><br>
      <img src="./output/head_wikidata_238_vs_239_lowest_bottom.png" width="100%">
    </td>
    <td align="center" width="50%">
      <strong>Tails</strong><br>
      <img src="./output/tail_wikidata_238_vs_239_lowest_bottom.png" width="100%">
    </td>
  </tr>
</table>

<!--Drift Line-->
### Line Graph Showing Drift
Plotting drift on the vertical axis on a sorted list of drift distances.

100 Epochs used in the training step.

<table>
  <tr>
    <th colspan="2" style="text-align:left;">Drift Distribution Line Plots: FB15k-237 vs FB15k-238</th>
  </tr>
  <tr>
    <td align="center" width="50%">
      <strong>Heads</strong><br>
      <img src="./img/topdrift/head_237_vs_238_line.png" width="100%">
    </td>
    <td align="center" width="50%">
      <strong>Tails</strong><br>
      <img src="./img/topdrift/tail_237_vs_238_line.png" width="100%">
    </td>
  </tr>
</table>

<br>

<table>
  <tr>
    <th colspan="2" style="text-align:left;">Drift Distribution Line Plots: FB15k-237 vs FB15k-239</th>
  </tr>
  <tr>
    <td align="center" width="50%">
      <strong>Heads</strong><br>
      <img src="./img/topdrift/head_237_vs_239_line.png" width="100%">
    </td>
    <td align="center" width="50%">
      <strong>Tails</strong><br>
      <img src="./img/topdrift/tail_237_vs_239_line.png" width="100%">
    </td>
  </tr>
</table>

<br>

<table>
  <tr>
    <th colspan="2" style="text-align:left;">Drift Distribution Line Plots: FB15k-238 vs FB15k-239</th>
  </tr>
  <tr>
    <td align="center" width="50%">
      <strong>Heads</strong><br>
      <img src="./img/topdrift/head_238_vs_239_line.png" width="100%">
    </td>
    <td align="center" width="50%">
      <strong>Tails</strong><br>
      <img src="./img/topdrift/tail_238_vs_239_line.png" width="100%">
    </td>
  </tr>
</table>

<!-- t-SNE and UMAP Visualizations -->
### t-SNE and UMAP visualizations

Visualizations for 237, 238, 239, (238-237), (239-238), (239-237), (T_237 -> M_238), (T_237 -> M_239). These visualizations can be found in img/tsneumap.

100 Epochs used in the training step.

<table>
  <tr>
    <th colspan="2" style="text-align:left;">fb15k237</th>
  </tr>
  <tr>
    <td align="center" width="50%">
      <strong>t-SNE</strong><br>
      <img src="./img/tsneumap/fb15k237_100epochs_tsne.png" width="100%">
    </td>
    <td align="center" width="50%">
      <strong>UMAP</strong><br>
      <img src="./img/tsneumap/fb15k237_100epochs_umap.png" width="100%">
    </td>
  </tr>
</table>

<table>
  <tr>
    <th colspan="2" style="text-align:left;">fb15k238</th>
  </tr>
  <tr>
    <td align="center" width="50%">
      <strong>t-SNE</strong><br>
      <img src="./img/tsneumap/fb15k238_100epochs_tsne.png" width="100%">
    </td>
    <td align="center" width="50%">
      <strong>UMAP</strong><br>
      <img src="./img/tsneumap/fb15k238_100epochs_umap.png" width="100%">
    </td>
  </tr>
</table>

<table>
  <tr>
    <th colspan="2" style="text-align:left;">fb15k238 minus fb15k237</th>
  </tr>
  <tr>
    <td align="center" width="50%">
      <strong>t-SNE</strong><br>
      <img src="./img/tsneumap/fb15k238_minus_237_100epochs_tsne.png" width="100%">
    </td>
    <td align="center" width="50%">
      <strong>UMAP</strong><br>
      <img src="./img/tsneumap/fb15k238_minus_237_100epochs_umap.png" width="100%">
    </td>
  </tr>
</table>

<table>
  <tr>
    <th colspan="2" style="text-align:left;">fb15k239</th>
  </tr>
  <tr>
    <td align="center" width="50%">
      <strong>t-SNE</strong><br>
      <img src="./img/tsneumap/fb15k239_100epochs_tsne.png" width="100%">
    </td>
    <td align="center" width="50%">
      <strong>UMAP</strong><br>
      <img src="./img/tsneumap/fb15k239_100epochs_umap.png" width="100%">
    </td>
  </tr>
</table>

<table>
  <tr>
    <th colspan="2" style="text-align:left;">fb15k239 minus fb15k237</th>
  </tr>
  <tr>
    <td align="center" width="50%">
      <strong>t-SNE</strong><br>
      <img src="./img/tsneumap/fb15k239_minus_237_100epochs_tsne.png" width="100%">
    </td>
    <td align="center" width="50%">
      <strong>UMAP</strong><br>
      <img src="./img/tsneumap/fb15k239_minus_237_100epochs_umap.png" width="100%">
    </td>
  </tr>
</table>

<table>
  <tr>
    <th colspan="2" style="text-align:left;">fb15k239 minus fb15k238</th>
  </tr>
  <tr>
    <td align="center" width="50%">
      <strong>t-SNE</strong><br>
      <img src="./img/tsneumap/fb15k239_minus_238_100epochs_tsne.png" width="100%">
    </td>
    <td align="center" width="50%">
      <strong>UMAP</strong><br>
      <img src="./img/tsneumap/fb15k239_minus_238_100epochs_umap.png" width="100%">
    </td>
  </tr>
</table>

<table>
  <tr>
    <th colspan="2" style="text-align:left;">fb15k237 (params from fb15k238)</th>
  </tr>
  <tr>
    <td align="center" width="50%">
      <strong>t-SNE</strong><br>
      <img src="./img/tsneumap/t237_m238_100epochs_tsne.png" width="100%">
    </td>
    <td align="center" width="50%">
      <strong>UMAP</strong><br>
      <img src="./img/tsneumap/t237_m238_100epochs_umap.png" width="100%">
    </td>
  </tr>
</table>

<table>
  <tr>
    <th colspan="2" style="text-align:left;">fb15k237 (params from fb15k239)</th>
  </tr>
  <tr>
    <td align="center" width="50%">
      <strong>t-SNE</strong><br>
      <img src="./img/tsneumap/t237_m239_100epochs_tsne.png" width="100%">
    </td>
    <td align="center" width="50%">
      <strong>UMAP</strong><br>
      <img src="./img/tsneumap/t237_m239_100epochs_umap.png" width="100%">
    </td>
  </tr>
</table>
