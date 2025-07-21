import matplotlib.pyplot as plt
import os
# Okay I could have done this a bit more cleanly but it was something of an afterthought
# Create output directory if it doesn't exist
os.makedirs("img", exist_ok=True)

# Epochs used in each comparison group
epochs = [0, 1, 5, 10, 25, 50, 100, 200]

# =====================
# Mean Drift Data
# =====================
head_mean = {
    'Δ₁': [0.376168, 0.548755, 0.680030, 0.745270, 0.824136, 0.866667, 0.899643, 0.932393],
    'Δ₂': [0.376168, 0.549280, 0.682184, 0.748099, 0.826599, 0.867281, 0.899607, 0.929965],
    'Δ₃': [0.000000, 0.237507, 0.439329, 0.560965, 0.701592, 0.770071, 0.814815, 0.849108]
}

relation_mean = {
    'Δ₁': [0.000000, 0.320820, 0.501356, 0.573533, 0.662607, 0.743669, 0.865690, 1.013992],
    'Δ₂': [0.000000, 0.324562, 0.497328, 0.577169, 0.660335, 0.740452, 0.861736, 1.014604],
    'Δ₃': [0.000000, 0.269317, 0.376040, 0.451789, 0.563823, 0.640996, 0.759481, 0.896226]
}

tail_mean = {
    'Δ₁': [0.369235, 0.532410, 0.643354, 0.700869, 0.773495, 0.816908, 0.863759, 0.910897],
    'Δ₂': [0.369235, 0.534347, 0.646124, 0.706011, 0.776617, 0.820152, 0.865926, 0.905163],
    'Δ₃': [0.000000, 0.234872, 0.411066, 0.524151, 0.648503, 0.718567, 0.769000, 0.811965]
}

# =====================
# Standard Deviation Data
# =====================
head_std = {
    'Δ₁': [0.624860, 0.501207, 0.313747, 0.225862, 0.151941, 0.136976, 0.131742, 0.127591],
    'Δ₂': [0.624860, 0.501335, 0.313614, 0.224192, 0.151162, 0.135929, 0.129738, 0.127397],
    'Δ₃': [0.000000, 0.028535, 0.064016, 0.081191, 0.107184, 0.124259, 0.130666, 0.130508]
}

relation_std = {
    'Δ₁': [0.000000, 0.084349, 0.161970, 0.192627, 0.249711, 0.313210, 0.402051, 0.530235],
    'Δ₂': [0.000000, 0.085230, 0.168527, 0.191033, 0.248810, 0.312770, 0.395032, 0.507872],
    'Δ₃': [0.000000, 0.036232, 0.096940, 0.130354, 0.198440, 0.256797, 0.32697, 0.413012]
}

tail_std = {
    'Δ₁': [0.620644, 0.483982, 0.307334, 0.238187, 0.182581, 0.168594, 0.146335, 0.151362],
    'Δ₂': [0.620644, 0.484941, 0.309677, 0.237694, 0.185076, 0.166163, 0.146631, 0.146783],
    'Δ₃': [0.000000, 0.028699, 0.082295, 0.107459, 0.143170, 0.145012, 0.147069, 0.144373]
}

def plot_and_save(metric_data, title, ylabel, filename):
    plt.figure(figsize=(10, 6))
    for label, values in metric_data.items():
        plt.plot(epochs, values, marker='o', label=label)
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"img/{filename}")
    plt.close()

# Save Mean Drift plots
plot_and_save(head_mean, "Mean Euclidean Drift Over Epochs (Head)", "Mean Drift", "head_mean_drift.png")
plot_and_save(relation_mean, "Mean Euclidean Drift Over Epochs (Relation)", "Mean Drift", "relation_mean_drift.png")
plot_and_save(tail_mean, "Mean Euclidean Drift Over Epochs (Tail)", "Mean Drift", "tail_mean_drift.png")

# Save Std Dev Drift plots
plot_and_save(head_std, "Standard Deviation of Drift Over Epochs (Head)", "Standard Deviation", "head_std_drift.png")
plot_and_save(relation_std, "Standard Deviation of Drift Over Epochs (Relation)", "Standard Deviation", "relation_std_drift.png")
plot_and_save(tail_std, "Standard Deviation of Drift Over Epochs (Tail)", "Standard Deviation", "tail_std_drift.png")
