import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams["figure.figsize"] = (8.0, 5.0)
plt.rcParams.update({'font.size': 16})

species = ("128","256",
           "512","1024",
           "2048","4096",)

weight_counts_original = {
    "Compute_Original"       : np.array([  36231.87, 16681.90, 7157.32, 1674.42, 758.22, 425.70 ]),
    "Communication_Original" : np.array([  622.03, 562.01, 295.14, 954.33, 363.69, 122.54 ]),
}

weight_counts_ops = {
    "Compute_OPS"       	 : np.array([  32321.4, 15330.30, 6701.26, 2816.73, 1185.57, 536.72 ]),
    "Communication_OPS" 	 : np.array([  1402.45, 768.01, 659.13, 312.42, 137.53, 112.84 ]),
}


# Bar settings
width = 0.3
#bar_gap = 0.05         # Gap between p1 and p2 in a group
bar_gap = 0.01
group_gap = 1.0        # Gap between different species

# Calculate bar positions
x_positions = []
xtick_positions = []
for i in range(len(species)):
    base = i * group_gap
    x_orig = base - (width / 2 + bar_gap / 2)
    x_ops = base + (width / 2 + bar_gap / 2)
    x_positions.append((x_orig, x_ops))
    xtick_positions.append(base)

# Colors
colors_original = ['navy', 'seagreen']
colors_ops = ['blue', 'lightgreen']
colors = ['#f05039', '#E57a77', '#eebab4', '#1f449c', '#3d65a5', '#7ca1cc', '#a8b6cc', '#8a5e00', '#ffc626', '#e6cf8e', '#444852', '#5c9e73', '#1a407d', '#9370DB', '#800080']

hatches_original = ['///', None]  # Compute_Original: hatched, Communication_Original: solid
hatches_ops = ['xxx', None]    # Compute_OPS: different hatched, Communication_OPS: solid
plt.rcParams['hatch.linewidth'] = 2.0

# Init plot
fig, ax = plt.subplots()
max_height = 0

# Plot bars for each core group
for i in range(len(species)):
    x_orig = x_positions[i][0]
    x_ops = x_positions[i][1]

    # Heights for each bar in the pair
    orig_compute = weight_counts_original["Compute_Original"][i]
    orig_comm = weight_counts_original["Communication_Original"][i]
    ops_compute = weight_counts_ops["Compute_OPS"][i]
    ops_comm = weight_counts_ops["Communication_OPS"][i]

     # Plot Original (p1)
    p1_compute = ax.bar(x_orig, orig_compute, width, color="white", hatch=hatches_original[0], edgecolor=colors[-1])
    p1_comm = ax.bar(x_orig, orig_comm, width, bottom=orig_compute, color=colors[-1], hatch=hatches_original[1], edgecolor=colors[-1])

    # Plot OPS (p2)
    p2_compute = ax.bar(x_ops, ops_compute, width, color="white", hatch=hatches_ops[0], edgecolor=colors[5])
    p2_comm = ax.bar(x_ops, ops_comm, width, bottom=ops_compute, color=colors[5], hatch=hatches_ops[1], edgecolor=colors[5])

    # Add outlines
    for bars in [p1_compute, p1_comm, p2_compute, p2_comm]:
        for bar in bars:
            ax.add_patch(Rectangle(
                (bar.get_x(), bar.get_y()),
                bar.get_width(),
                bar.get_height(),
#                edgecolor='black',
                linewidth=0.0,
                fill=False
            ))

    # Calculate the total heights
    total_orig = orig_compute + orig_comm
    total_ops = ops_compute + ops_comm


    if total_orig < total_ops:
        label_offset_original = total_orig + total_orig * 0.01
        label_offset_ops = total_ops + total_ops * 0.04
    else:
        label_offset_original = total_orig + total_orig * 0.04
        label_offset_ops = total_ops + total_ops * 0.01

    # Add total labels for each p1 and p2 on top in black color
#    ax.text(x_orig, label_offset_original, f'{total_orig:.0f}', ha='center', va='bottom', color='black', fontsize=11)#, fontweight='bold')
#    ax.text(x_ops, label_offset_ops, f'{total_ops:.0f}', ha='center', va='bottom', color='black', fontsize=11)#, fontweight='bold')

    # Track max height
    max_height = max(max_height, total_orig, total_ops)

# Axes setup
ax.set_xticks(xtick_positions)
ax.set_xticklabels(species)
ax.set_ylabel('Runtime (s)', fontsize=20)
#ax.set_xlabel("LUMI-C cores", fontsize=16, labelpad=15)
ax.set_xlabel("Cores", fontsize=20, labelpad=0)
ax.set_ylim(0, 40000)
ax.grid(axis='y', linestyle=':')

# Legend with correct color mapping
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="white", hatch='///', label='Original - Compute', edgecolor=colors[-1]),
    Patch(facecolor=colors[-1], label='Original - Communication', edgecolor=colors[-1]),
    Patch(facecolor="white", hatch='xxx', label='OPS - Compute', edgecolor=colors[5]),
    Patch(facecolor=colors[5], label='OPS - Communication', edgecolor=colors[5]),
]
ax.legend(handles=legend_elements, loc="upper right", edgecolor='black', fontsize=16, ncol=1)

# Ticks
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)

# Layout and save
fig.tight_layout()
fig.subplots_adjust(bottom=0.2)
plt.savefig("senga_multi-node_3dh2flame_lumi-c_cpuscaling.pdf", format="pdf", bbox_inches="tight")
# plt.show()

