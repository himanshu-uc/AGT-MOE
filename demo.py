import pandas as pd

df = pd.read_csv('./alpha_2_5.csv')
print(df.columns)

import pandas as pd
import matplotlib.pyplot as plt

# Assume df is already loaded
# df = pd.read_csv("your_file.csv")

# Metrics to plot (y-axis)
metric_cols = ['loss_train', 'loss_val', 'Expert Utilization', 'Load Imbalance Ratio',
               'Routing Concentration', 'Expert Load Variation', 'Routing Probability CV']

# Average across seeds for each (temperature, iter)
grouped_df = df.groupby(['temperature', 'iter'])[metric_cols].mean().reset_index()

# Grid dimensions (adjust as needed)
n_cols = 3
n_rows = -(-len(metric_cols) // n_cols)  # Ceiling division

# Create figure and axes
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharex=True)
axes = axes.flatten()

# Plot each metric
for idx, metric in enumerate(metric_cols):
    ax = axes[idx]
    for temp in sorted(df['temperature'].unique()):
        temp_df = grouped_df[grouped_df['temperature'] == temp].sort_values('iter')
        ax.plot(temp_df['iter'], temp_df[metric], label=f'temp={temp}')

    ax.set_title(metric)
    ax.set_xlabel("iter")
    ax.set_ylabel(metric)
    ax.grid(True)
    ax.legend(fontsize='small')

# Hide unused subplots if any
for j in range(len(metric_cols), len(axes)):
    fig.delaxes(axes[j])

fig.suptitle("Metrics vs Iter (Averaged across Seeds)", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()