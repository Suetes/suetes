#!/usr/bin/env python3
"""
Plots scaling performance metrics for the Suêtes GMD manuscript.
Ingests benchmark CSV files and outputs a publication-quality figure.
"""
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

output_dir = "output/plots/benchmarks"
os.makedirs(output_dir, exist_ok=True)

def load_benchmark_data():
    """Finds and combines all massive benchmark CSVs in the directory."""
    csv_files = glob.glob("output/benchmark_bubble_scaling_*.csv") + glob.glob("benchmark_results_massive_*.csv")
    if not csv_files:
        raise FileNotFoundError("No benchmark CSV files found. Please run the benchmark script first.")
    
    # Take the most recent or combine if multiple GPUs exist
    print(f"Found benchmark data: {csv_files}")
    dfs = []
    for f in csv_files:
        df_temp = pd.read_csv(f)
        # Ensure dt_multiplier exists for backward compatibility
        if 'dt_multiplier' not in df_temp.columns:
            if 'dt' in df_temp.columns:
                df_temp['dt_multiplier'] = (df_temp['dt'] / 2.5).round()
            else:
                df_temp['dt_multiplier'] = np.where(df_temp['core'] == 'sisl', 10.0, 1.0)
        dfs.append(df_temp)
        
    return pd.concat(dfs).drop_duplicates(subset=['core', 'N', 'dt_multiplier'])

def main():
    try:
        df = load_benchmark_data()
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    # Separate by dynamical core type and timestep configuration
    se_data = df[(df['core'] == 'split-explicit') & (df['dt_multiplier'] == 1.0)].sort_values('N')
    sisl_1x_data = df[(df['core'] == 'sisl') & (df['dt_multiplier'] == 1.0)].sort_values('N')
    # Determine large dt multiplier (10.0 or 6.0)
    large_dt_mult = 10.0 if (df['dt_multiplier'] == 10.0).any() else 6.0
    sisl_large_data = df[(df['core'] == 'sisl') & (df['dt_multiplier'] == large_dt_mult)].sort_values('N')
    label_a_large = r'SISL ($\Delta t = 25.0$s)' if large_dt_mult == 10.0 else r'SISL ($\Delta t = 15.0$s)'
    label_bc_large = r'SISL ($10\times$ dt)' if large_dt_mult == 10.0 else r'SISL ($6\times$ dt)'

    has_sisl_1x = not sisl_1x_data.empty

    # Setup the plot layout (1 row, 3 columns for GMD formatting)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    
    # Common styling configurations
    marker_size = 8
    line_width = 2.2
    grid_style = dict(ls='--', alpha=0.5, which='both')

    # ---------------------------------------------------------
    # PANEL A: Simulated Years Per Day (SYPD) - Throughput
    # ---------------------------------------------------------
    ax = axes[0]
    ax.semilogy(se_data['N'], se_data['sypd'], 'o-', color='#e41a1c', 
            lw=line_width, ms=marker_size, label=r'Split-explicit ($\Delta t = 2.5$s)')
    if has_sisl_1x:
        ax.semilogy(sisl_1x_data['N'], sisl_1x_data['sypd'], 'd-', color='#4daf4a', 
                lw=line_width, ms=marker_size, label=r'SISL ($\Delta t = 2.5$s)')
    ax.semilogy(sisl_large_data['N'], sisl_large_data['sypd'], 's-', color='#377eb8', 
            lw=line_width, ms=marker_size, label=label_a_large)
    
    ax.set_title('(a) Simulation throughput', fontsize=14, pad=10)
    ax.set_xlabel('Grid dimension ($N \\times N \\times N$)', fontsize=12)
    ax.set_ylabel('Throughput (SYPD)', fontsize=12)
    ax.set_xticks(df['N'].unique())
    ax.grid(True, **grid_style)
    ax.legend(fontsize=10, loc='upper right')

    # ---------------------------------------------------------
    # PANEL B: Step Latency (Compute Scaling)
    # ---------------------------------------------------------
    ax = axes[1]
    ax.loglog(se_data['N'], se_data['ms_per_step'], 'o-', color='#e41a1c', lw=line_width, ms=marker_size, label='Split-explicit')
    if has_sisl_1x:
        ax.loglog(sisl_1x_data['N'], sisl_1x_data['ms_per_step'], 'd-', color='#4daf4a', lw=line_width, ms=marker_size, label=r'SISL ($1\times$ dt)')
    ax.loglog(sisl_large_data['N'], sisl_large_data['ms_per_step'], 's-', color='#377eb8', lw=line_width, ms=marker_size, label=label_bc_large)
    
    # Add an ideal O(N^3) line to show where compute saturation happens
    n_vals = np.array(se_data['N'])
    if len(n_vals) > 1:
        # Reference line aligned to the final compute-bound point of Split-Explicit
        ref_line = se_data['ms_per_step'].iloc[-1] * (n_vals / n_vals[-1])**3
        ax.loglog(n_vals, ref_line, 'k--', alpha=0.7, label=r'Ideal $\mathcal{O}(N^3)$ compute')
    
    ax.set_title('(b) Compute scaling', fontsize=14, pad=10)
    ax.set_xlabel('Grid dimension ($N \\times N \\times N$)', fontsize=12)
    ax.set_ylabel('Execution latency (ms/step)', fontsize=12)
    ax.set_xticks(df['N'].unique())
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.minorticks_off()
    ax.grid(True, **grid_style)
    ax.legend(fontsize=10, loc='upper left')

    # ---------------------------------------------------------
    # PANEL C: Peak VRAM Allocation
    # ---------------------------------------------------------
    ax = axes[2]
    ax.loglog(se_data['N'], se_data['vram_mb'], 'o-', color='#e41a1c', lw=line_width, ms=marker_size, label='Split-explicit')
    if has_sisl_1x:
        ax.loglog(sisl_1x_data['N'], sisl_1x_data['vram_mb'], 'd-', color='#4daf4a', lw=line_width, ms=marker_size, label=r'SISL ($1\times$ dt)')
    ax.loglog(sisl_large_data['N'], sisl_large_data['vram_mb'], 's-', color='#377eb8', lw=line_width, ms=marker_size, label=label_bc_large)
    
    ax.set_title('(c) Peak VRAM allocation', fontsize=14, pad=10)
    ax.set_xlabel('Grid dimension ($N \\times N \\times N$)', fontsize=12)
    ax.set_ylabel('GPU memory used (MiB)', fontsize=12)
    ax.set_xticks(df['N'].unique())
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.minorticks_off()
    ax.grid(True, **grid_style)
    ax.legend(fontsize=10, loc='upper left')

    # Save final high-res figure for GMD submission to centralized output
    out_path = f"{output_dir}/suetes_scaling_metrics.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Figure saved to: {out_path}")

if __name__ == "__main__":
    main()
