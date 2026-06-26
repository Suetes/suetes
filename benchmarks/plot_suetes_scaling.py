#!/usr/bin/env python3
"""
Plots scaling performance metrics for the Suêtes GMD manuscript.
Ingests benchmark CSV files and outputs a publication-quality figure.
"""
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_benchmark_data():
    """Finds and combines all massive benchmark CSVs in the directory."""
    csv_files = glob.glob("benchmark_results_massive_*.csv")
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
                df_temp['dt_multiplier'] = np.where(df_temp['core'] == 'sisl', 6.0, 1.0)
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
    sisl_6x_data = df[(df['core'] == 'sisl') & (df['dt_multiplier'] == 6.0)].sort_values('N')

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
    ax.plot(se_data['N'], se_data['sypd'], 'o-', color='#e41a1c', 
            lw=line_width, ms=marker_size, label=r'Split-Explicit ($\Delta t = 2.5$s)')
    if has_sisl_1x:
        ax.plot(sisl_1x_data['N'], sisl_1x_data['sypd'], 'd-', color='#4daf4a', 
                lw=line_width, ms=marker_size, label=r'SISL ($\Delta t = 2.5$s)')
    ax.plot(sisl_6x_data['N'], sisl_6x_data['sypd'], 's-', color='#377eb8', 
            lw=line_width, ms=marker_size, label=r'SISL ($\Delta t = 15.0$s)')
    
    ax.set_title('(a) Simulation Throughput', fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel('Grid Dimension ($N \\times N \\times N$)', fontsize=12)
    ax.set_ylabel('Throughput (SYPD)', fontsize=12)
    ax.set_xticks(df['N'].unique())
    ax.grid(True, **grid_style)
    ax.legend(fontsize=10, loc='upper right')

    # ---------------------------------------------------------
    # PANEL B: Step Latency (Compute Scaling)
    # ---------------------------------------------------------
    ax = axes[1]
    ax.loglog(se_data['N'], se_data['ms_per_step'], 'o-', color='#e41a1c', lw=line_width, ms=marker_size, label='Split-Explicit')
    if has_sisl_1x:
        ax.loglog(sisl_1x_data['N'], sisl_1x_data['ms_per_step'], 'd-', color='#4daf4a', lw=line_width, ms=marker_size, label=r'SISL ($1\times$ dt)')
    ax.loglog(sisl_6x_data['N'], sisl_6x_data['ms_per_step'], 's-', color='#377eb8', lw=line_width, ms=marker_size, label=r'SISL ($6\times$ dt)')
    
    # Add an ideal O(N^3) line to show where compute saturation happens
    n_vals = np.array(se_data['N'])
    if len(n_vals) > 1:
        # Reference line aligned to the final compute-bound point of Split-Explicit
        ref_line = se_data['ms_per_step'].iloc[-1] * (n_vals / n_vals[-1])**3
        ax.loglog(n_vals, ref_line, 'k--', alpha=0.7, label=r'Ideal $\mathcal{O}(N^3)$ Compute')
    
    ax.set_title('(b) Compute Scaling Law', fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel('Grid Dimension ($N \\times N \\times N$)', fontsize=12)
    ax.set_ylabel('Execution Latency (ms/step)', fontsize=12)
    ax.set_xticks(df['N'].unique())
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.grid(True, **grid_style)
    ax.legend(fontsize=10, loc='upper left')

    # ---------------------------------------------------------
    # PANEL C: Peak VRAM Allocation
    # ---------------------------------------------------------
    ax = axes[2]
    ax.plot(se_data['N'], se_data['vram_mb'], 'o-', color='#e41a1c', lw=line_width, ms=marker_size, label='Split-Explicit')
    if has_sisl_1x:
        ax.plot(sisl_1x_data['N'], sisl_1x_data['vram_mb'], 'd-', color='#4daf4a', lw=line_width, ms=marker_size, label=r'SISL ($1\times$ dt)')
    ax.plot(sisl_6x_data['N'], sisl_6x_data['vram_mb'], 's-', color='#377eb8', lw=line_width, ms=marker_size, label=r'SISL ($6\times$ dt)')
    
    ax.set_title('(c) Peak VRAM Allocation', fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel('Grid Dimension ($N \\times N \\times N$)', fontsize=12)
    ax.set_ylabel('GPU Memory Used (MB)', fontsize=12)
    ax.set_xticks(df['N'].unique())
    ax.grid(True, **grid_style)
    ax.legend(fontsize=10, loc='upper left')

    # Save final high-res figure for GMD submission to centralized output
    out_path = "output/plots/suetes_scaling_metrics.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Publication figure saved cleanly to: {out_path}")

if __name__ == "__main__":
    main()
