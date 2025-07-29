import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams["figure.figsize"] = (10.0, 6.5)
plt.rcParams.update({'font.size': 18})

fig = plt.figure()
gs = gridspec.GridSpec(1, 1)
fig.set_tight_layout(True)
ax0 = fig.add_subplot(gs[0])
fig.tight_layout()
ax0.set_ylim([8, 30000])
ax0.set_xlim([0.01, 100])
markers2=['o', '*', 's', '^', 'X', 'D', 'v']
#markers2=['x', '+', '^', 's', 'o']
senga_opp_names = ["TEMPER", "DIFFUSIVITY", "THIRD_BODY", "DFBYDZ", "GIBBS"]
senga_opp_x = [0.12	       , 4.08	    , 0.11	    , 1.24	    , 3.01        ]     # AI - these are from BEDE
senga_opp_y = [127         , 4640       , 123       , 707       , 3102        ]

for x, y, marker_type, names in zip(senga_opp_x, senga_opp_y, markers2, senga_opp_names):
    markersize1=18
    if marker_type == '*':
        markersize1=20
    ax0.plot(x, y, marker=marker_type, markersize=markersize1, linestyle="None", label=names, markerfacecolor='None', markeredgecolor='darkblue', markeredgewidth=3)

max_gflops_dp = 7120.55
max_gflops_sp = 14128.51
max_gflops = max(max_gflops_sp, max_gflops_dp)
l1_bandwidth = 3510.55
dram_bandwidth = 1496.06
l1_ai = max_gflops / l1_bandwidth
dram_ai = max_gflops / dram_bandwidth
x_max_gflops_sp = [max_gflops_sp / l1_bandwidth, 100]
y_max_gflops_sp = [max_gflops_sp, max_gflops_sp]
x_max_gflops_dp = [max_gflops_dp / l1_bandwidth, 100]
y_max_gflops_dp = [max_gflops_dp, max_gflops_dp]
x_l1_line = [0, l1_ai]
y_l1_line = [0, max_gflops]
x_dram_line = [0, dram_ai]
y_dram_line = [0, max_gflops]
line_colour = '#5A5A5A'
line_width = 1.5
ax0.plot(x_max_gflops_sp, y_max_gflops_sp, c=line_colour,linewidth=line_width, zorder=1)
ax0.plot(x_max_gflops_dp, y_max_gflops_dp, c=line_colour,linewidth=line_width, zorder=1)
ax0.plot(x_l1_line, y_l1_line, c=line_colour,linewidth=line_width, zorder=1)
ax0.plot(x_dram_line, y_dram_line, c=line_colour,linewidth=line_width, zorder=1)
x_dotted_sp = [0.001, l1_ai]
x_dotted_dp = [0.001, max_gflops_dp / l1_bandwidth]
x_dotted_sp = [0.001, max_gflops_sp / l1_bandwidth]
ax0.plot(x_dotted_sp, y_max_gflops_sp, c=line_colour,linewidth=1.0, linestyle='--', zorder=1)
ax0.annotate(str(max_gflops_sp) + ' GFLOPS/s SP Max', (0.012, max_gflops_sp + 800), fontsize=20, zorder=0, color='darkblue')
ax0.plot(x_dotted_dp, y_max_gflops_dp, c=line_colour,linewidth=1.0, linestyle='--', zorder=1)
ax0.annotate(str(max_gflops_dp) + ' GFLOPS/s DP Max', (0.012, max_gflops_dp - 2000), fontsize=20, zorder=0, color='darkblue')
dram_diag_x = 0.0025
diag_x = 0.0025
ax0.annotate('L1 - ' + str(l1_bandwidth) + ' GB/s', (0.012, 55), fontsize=20, rotation=36.5, zorder=0, color='darkblue')
ax0.annotate('DRAM - ' + str(dram_bandwidth) + ' GB/s', (0.012, 23), fontsize=20, rotation=36.5, zorder=0, color='darkblue')
props = dict(boxstyle='round', facecolor='white', alpha=1.0)

ax0.set_xscale('log')
ax0.set_yscale('log')
ax0.grid(which='major', axis='both', linestyle=':', linewidth=1.0)

# Set custom ticks
ax0.set_xticks([0.01, 0.1, 1, 10, 100])
ax0.set_yticks([10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 30000])

# Format ticks
ax0.xaxis.set_major_formatter(ScalarFormatter())
ax0.yaxis.set_major_formatter(ScalarFormatter())
ax0.tick_params(axis='both', which='major', labelsize=20)

# Legends
handles, labels = ax0.get_legend_handles_labels()
ax0.legend(handles, labels, loc='lower right', ncol=1, facecolor='w', framealpha=1, edgecolor='black', prop={'size': 18})

# X and Y - Axis labels
ax0.set_xlabel('Arithmetic Intensity (FLOPs/Byte)', fontsize=22)
ax0.set_ylabel('Performance (GFLOPs/s)', fontsize=22)

fig.tight_layout()

plt.savefig("senga_roofline_tursa.pdf", format="pdf", bbox_inches="tight")


#plt.show()
