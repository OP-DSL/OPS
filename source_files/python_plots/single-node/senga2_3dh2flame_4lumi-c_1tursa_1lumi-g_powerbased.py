import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams["figure.figsize"] = (8.0, 5.0)
plt.rcParams.update({'font.size': 16})

species = ("LUMI-C\n(512,4)\nOriginal","LUMI-C\n(512,4)\nOPS-CPU",
           "TURSA\n(4,1)\nCUDA FORT.","TURSA\n(4,1)\nCUDA C",
           "LUMI-G\n(8,1)\nHIP C",)

# senga2_3dh2flame_4lumi-c_1tursa_1lumi-g_powerbased.py
weight_counts = {
    "Compute"       : np.array([ 7157.32, 6701.26, 2248.81, 1873.65, 1493.53 ]),
    "Communication" : np.array([ 295.14,  659.13, 142.38,  88.13,  164.01  ]),
}

# Bar settings
width = 0.5

# Calculate bar positions
x_positions = []
xtick_positions = []
for i in range(len(species)):
    base = i
    x_orig = base
    x_positions.append(x_orig)
    xtick_positions.append(base)

# Colors
#colors = ['#f05039', '#E57a77', '#eebab4', '#1f449c', '#3d65a5', '#7ca1cc', '#a8b6cc', '#8a5e00', '#ffc626', '#e6cf8e', '#444852', '#5c9e73', '#1a407d', '#9370DB', '#800080']
#colors = ['#f05039', '#E57a77', '#eebab4', '#1f449c', '#3d65a5', '#7ca1cc', '#a8b6cc', '#8a5e00', '#ffc626', '#60b858', '#444852', '#5c9e73', '#1a407d', '#9370DB', '#800080']
colors = ['#a31c20', '#E57a77', '#eebab4', '#1f449c', '#3d65a5', '#7ca1cc', '#a8b6cc', '#8a5e00', '#ffc626', '#60b858', '#444852', '#5c9e73', '#1a407d', '#9370DB', '#800080']

bar_colors = [colors[-1], colors[5], colors[9], colors[9], colors[0]]

hatches_compute = ['//', 'xx', '\\\\', '\\\\', '+']  # Compute_Original: hatched, Communication_Original: solid
hatches_comm = [None, None, None, None, None]
plt.rcParams['hatch.linewidth'] = 3.0

# Init plot
fig, ax = plt.subplots()
max_height = 0

# Plot bars for each core group
for i in range(len(species)):
    x_orig = i

    # Heights for each bar in the pair
    orig_compute = weight_counts["Compute"][i]
    orig_comm = weight_counts["Communication"][i]

     # Plot Original (p1)
    p1_compute = ax.bar(x_orig, orig_compute, width, color="white", hatch=hatches_compute[i], edgecolor=bar_colors[i])
    p1_comm = ax.bar(x_orig, orig_comm, width, bottom=orig_compute, color=bar_colors[i], hatch=hatches_comm[i], edgecolor=bar_colors[i])

    # Add outlines
    for bars in [p1_compute, p1_comm]:
        for bar in bars:
            ax.add_patch(Rectangle(
                (bar.get_x(), bar.get_y()),
                bar.get_width(),
                bar.get_height(),
                edgecolor='black',
                linewidth=1.0,
                fill=False
            ))

    # Calculate the total heights
    total_orig = orig_compute + orig_comm

    label_offset = total_orig + total_orig * 0.01

    # Add total labels for each p1 and p2 on top in black color
    ax.text(x_orig, label_offset, f'{total_orig:.0f}', ha='center', va='bottom', color='black', fontsize=14)#, fontweight='bold')

    # Track max height
    max_height = max(max_height, total_orig)

# Axes setup
ax.set_xticks(xtick_positions)
ax.set_xticklabels(species)
ax.set_ylabel('Runtime (s)', fontsize=21)
ax.set_xlabel("System, (Ranks,nodes), Parallelization", fontsize=21, labelpad=1)
ax.set_ylim(0, 8000)
ax.grid(axis='y', linestyle=':')

# Legend with correct color mapping
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="white", hatch='///', label='Compute', edgecolor="black"),
    Patch(facecolor="black", label='Communication', edgecolor="black"),
]
ax.legend(handles=legend_elements, loc="upper right", edgecolor='black', fontsize=16, ncol=1)

# Ticks
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)

# Layout and save
fig.tight_layout()
fig.subplots_adjust(bottom=0.2)
plt.savefig("senga2_3dh2flame_4lumi-c_1tursa_1lumi-g_powerbased.pdf", format="pdf", bbox_inches="tight")
# plt.show()

