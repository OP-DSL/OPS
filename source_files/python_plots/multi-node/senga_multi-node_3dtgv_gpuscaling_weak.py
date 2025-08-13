import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams["figure.figsize"] = (8.0, 5.0)
plt.rcParams.update({'font.size': 16})

species = ("1","2",
           "4","8",
           "16","32",
           "64","128",
           "256",)

weight_counts_cuda = {
    "Compute_cuda"       : np.array([  2013.04, 1992.52, 1986.96, 1979.08, 1972.64, 0, 0, 0, 0 ]),
    "Communication_cuda" : np.array([  79.36,   208.34,  299.26,  314.42,  338.85,  0, 0, 0, 0 ]),
}

weight_counts_hip = {
    "Compute_hip"       	 : np.array([  1633.12, 1628.30, 1634.54, 1628.92, 1629.17, 1630.37, 1630.90, 1630.15, 1631.05 ]),
    "Communication_hip" 	 : np.array([  172.47,  224.06,  224.59,  323.73,  311.10,  314.81,  309.83,  315.29, 312.18 ]),
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
    x_hip = base + (width / 2 + bar_gap / 2)
    x_positions.append((x_orig, x_hip))
    xtick_positions.append(base)

# Colors
colors_cuda = ['navy', 'seagreen']
colors_hip = ['blue', 'lightgreen']
colors = ['#a31c20', '#E57a77', '#eebab4', '#1f449c', '#3d65a5', '#7ca1cc', '#a8b6cc', '#8a5e00', '#ffc626', '#60b858', '#444852', '#5c9e73', '#1a407d', '#9370DB', '#800080']

bar_colors = [colors[-1], colors[5], colors[9], colors[9], colors[0]]

hatches_cuda = ['\\\\', None]  # Compute_cuda: hatched, Communication_cuda: solid
hatches_hip = ['+', None]    # Compute_hip: different hatched, Communication_hip: solid
plt.rcParams['hatch.linewidth'] = 3.0

# Init plot
fig, ax = plt.subplots()
max_height = 0

# Plot bars for each core group
for i in range(len(species)):
    x_orig = x_positions[i][0]
    x_hip = x_positions[i][1]

    # Heights for each bar in the pair
    orig_compute = weight_counts_cuda["Compute_cuda"][i]
    orig_comm = weight_counts_cuda["Communication_cuda"][i]
    ops_compute = weight_counts_hip["Compute_hip"][i]
    ops_comm = weight_counts_hip["Communication_hip"][i]

     # Plot Original (p1)
    p1_compute = ax.bar(x_orig, orig_compute, width, color="white", hatch=hatches_cuda[0], edgecolor=bar_colors[3])
    p1_comm = ax.bar(x_orig, orig_comm, width, bottom=orig_compute, color=bar_colors[3], hatch=hatches_cuda[1], edgecolor=bar_colors[3])

    # Plot OPS (p2)
    p2_compute = ax.bar(x_hip, ops_compute, width, color="white", hatch=hatches_hip[0], edgecolor=bar_colors[4])
    p2_comm = ax.bar(x_hip, ops_comm, width, bottom=ops_compute, color=bar_colors[4], hatch=hatches_hip[1], edgecolor=bar_colors[4])

    # Add outlines
    for bars in [p1_compute, p1_comm, p2_compute, p2_comm]:
        for bar in bars:
            ax.add_patch(Rectangle(
                (bar.get_x(), bar.get_y()),
                bar.get_width(),
                bar.get_height(),
#                edgecolor='black',
                linewidth=0.5,
                fill=False
            ))

    # Calculate the total heights
    total_orig = orig_compute + orig_comm
    total_hip = ops_compute + ops_comm


    if total_orig < total_hip:
        label_offset_cuda = total_orig + total_orig * 0.01
        label_offset_hip = total_hip + total_hip * 0.04
    else:
        label_offset_cuda = total_orig + total_orig * 0.04
        label_offset_hip = total_hip + total_hip * 0.01

    # Add total labels for each p1 and p2 on top in black color
#    ax.text(x_orig, label_offset_cuda, f'{total_orig:.0f}', ha='center', va='bottom', color='black', fontsize=11)#, fontweight='bold')
#    ax.text(x_hip, label_offset_hip, f'{total_hip:.0f}', ha='center', va='bottom', color='black', fontsize=11)#, fontweight='bold')

    # Track max height
    max_height = max(max_height, total_orig, total_hip)

# Axes setup
ax.set_xticks(xtick_positions)
ax.set_xticklabels(species)
ax.set_ylabel('Runtime (s)', fontsize=20)
#ax.set_xlabel("LUMI-C cores", fontsize=16, labelpad=15)
ax.set_xlabel("Nodes", fontsize=20, labelpad=0)
ax.set_ylim(0, 3000)
ax.grid(axis='y', linestyle=':')

# Legend with correct color mapping
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="white", hatch='\\\\', label='CUDA C - Compute', edgecolor=bar_colors[3]),
    Patch(facecolor=bar_colors[3], label='CUDA C - Communication', edgecolor=bar_colors[3]),
    Patch(facecolor="white", hatch='+', label='HIP C - Compute', edgecolor=bar_colors[4]),
    Patch(facecolor=bar_colors[4], label='HIP C - Communication', edgecolor=bar_colors[4]),
]
ax.legend(handles=legend_elements, loc="upper right", edgecolor='black', fontsize=14, ncol=2)

# Ticks
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)

# Layout and save
fig.tight_layout()
fig.subplots_adjust(bottom=0.2)
plt.savefig("senga_multi-node_3dtgv_gpuscaling_weak.pdf", format="pdf", bbox_inches="tight")
# plt.show()

