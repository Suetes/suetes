import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal

def plot_point_timeseries(times_hours, suetes_values, era5_values,
                          location_name, units='K', save_path=None, ml_values=None):
    """
    Plot a single-cell timeseries comparing Suetes (and optionally ML-Suetes) against the ERA5 driver.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    if ml_values is not None:
        ax.plot(times_hours, suetes_values, 'b-',  linewidth=2, alpha=0.4, label='Suetes (Baseline)')
        ax.plot(times_hours, ml_values, 'g-', linewidth=2, label='Suetes (ML Corrected)')
    else:
        ax.plot(times_hours, suetes_values, 'b-',  linewidth=2, label='Suetes')
        
    ax.plot(times_hours, era5_values,   'r--', linewidth=2, label='ERA5')
    
    # Dynamically determine the variable name based on the specified units
    if units in ['km/h', 'm/s']:
        var_name = "Wind Speed"
    elif units in ['K', 'C', 'deg C']:
        var_name = "Near-surface Temperature"
    else:
        var_name = "Variable"
    
    ax.set_xlabel('Time [hours]')
    ax.set_ylabel(f'{var_name} [{units}]')
    ax.set_title(f'{var_name} over {location_name}')
    ax.grid(True, ls='--', alpha=0.5)
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()

def plot_energy_spectrum(grid, state_or_states, variable='w', z_idx=5, sponge_depth=30, 
                         ax=None, color='b', label=None, plot_reference_slopes=True, save_path=None):
    r"""
    Computes and plots the 1D spatial power spectrum using FFT.

    A fundamental check for the physical accuracy of the dynamical core's 
    dissipation schemes is comparing the resolved kinetic energy spectrum 
    against the theoretical Kolmogorov isotropic turbulence cascade:

    $$ E(k) \propto k^{-5/3} $$

    Accepts either a single state dict (one snapshot) or a list of state
    dicts (time-averages the spectra across snapshots). Time averaging
    reduces the per-wavenumber variance of the spectral estimate.
    """
    # Accept a single state dict for backward compatibility, or a list/tuple
    # of state dicts for time-averaging.
    if isinstance(state_or_states, dict):
        states = [state_or_states]
    else:
        states = list(state_or_states)

    if not states:
        raise ValueError("The states list provided for spectral analysis is empty.")

    dx = grid.dx
    spectra = []
    
    # Initialize data_inner in the broader scope
    data_inner = None 
    
    for state in states:
        data_2d = state[variable][:, :, z_idx]
        if sponge_depth > 0:
            data_inner = data_2d[sponge_depth:-sponge_depth, sponge_depth:-sponge_depth]
        else:
            data_inner = data_2d

        for row in data_inner.T:
            row_windowed = signal.detrend(row) * np.hanning(len(row))
            spectra.append(np.abs(np.fft.rfft(row_windowed))**2)

    avg_power = np.mean(spectra, axis=0)[1:]
    k = np.fft.rfftfreq(data_inner.shape[0], d=dx)[1:]
    
    show_plot = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        show_plot = True
        
    line_label = label if label else f"Spectrum ({variable})"
    ax.loglog(k, avg_power, color=color, label=line_label, linewidth=2, alpha=0.8)
    
    if plot_reference_slopes:
        k_transition = 1.0 / 400e3
        split = int(np.clip(np.searchsorted(k, k_transition), 1, len(k) - 1))

        ref_k_low = k[: split + 1]
        i_low = max(1, split // 2)
        ax.loglog(ref_k_low, avg_power[i_low] * (ref_k_low / k[i_low]) ** (-3),
                  'k:', label="$k^{-3}$ slope")

        ref_k_high = k[split:]
        i_high = split + (len(k) - split) // 2
        ax.loglog(ref_k_high, avg_power[i_high] * (ref_k_high / k[i_high]) ** (-5/3),
                  'k--', label="$k^{-5/3}$ slope")
        
        ax.axvline(x=1.0/(2*dx), color='r', linestyle=':', label=rf'$2\Delta x$ ({2*dx/1000:.1f} km)')
        ax.axvline(x=1.0/(6*dx), color='orange', linestyle=':', label=rf'$6\Delta x$ ({6*dx/1000:.1f} km)')

        def safe_inv(x):
            x_arr = np.array(x, dtype=float)
            return np.divide(1.0, x_arr, out=np.full_like(x_arr, np.inf), where=(x_arr != 0))
            
        secax = ax.secondary_xaxis('top', functions=(safe_inv, safe_inv))
        secax.set_xlabel('Wavelength [m]')
        secax.set_xticks([1e5, 5e4, 2e4, 1.2e4])
        secax.set_xticklabels(['100km', '50km', '20km', '12km'])

    if show_plot:
        title_suffix = f" (averaged over {len(states)} snapshots)" if len(states) > 1 else ""
        ax.set_title(f"Power spectrum: {variable} at level {z_idx}{title_suffix}")
        ax.set_xlabel("Wavenumber $k$ [m⁻¹]")
        ax.set_ylabel("Spectral power density")
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.legend()
        plt.tight_layout()
        
        if save_path: plt.savefig(save_path, dpi=200)
        else: plt.show()
        plt.close()
