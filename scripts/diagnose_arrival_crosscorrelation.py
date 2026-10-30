"""Compare les pics et les retards estimés par corrélation croisée."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import correlate, correlation_lags

from seismic.model.velocity_model import VelocityModel
from seismic.wave import simulate_acoustic_wave


def main() -> None:
    # Même expérience homogène que dans la validation précédente.
    dx_m = 2.0
    dz_m = 2.0
    velocity_m_s = 1500.0

    x_m = np.arange(0.0, 240.0 + dx_m, dx_m)
    z_m = np.arange(0.0, 120.0 + dz_m, dz_m)

    velocity = np.full(
        (z_m.size, x_m.size),
        velocity_m_s,
        dtype=np.float64,
    )

    model = VelocityModel(
        x_m=x_m,
        z_m=z_m,
        velocity_m_s=velocity,
        anomaly_mask=np.zeros_like(velocity, dtype=bool),
    )

    source_x_m = 60.0
    source_z_m = 50.0
    receiver_x_m = np.array(
        [80.0, 100.0, 120.0, 140.0, 160.0],
        dtype=np.float64,
    )
    receiver_z_m = 50.0

    frequency_hz = 25.0
    dt_s = 4.0e-4
    duration_s = 0.16

    result = simulate_acoustic_wave(
        model,
        duration_s=duration_s,
        dt_s=dt_s,
        source_x_m=source_x_m,
        source_z_m=source_z_m,
        source_frequency_hz=frequency_hz,
        receiver_x_m=receiver_x_m,
        receiver_z_m=receiver_z_m,
        sponge_width_m=40.0,
        sponge_strength_s_inv=80.0,
        snapshot_stride=25,
    )

    times_s = result.sample_times_s
    distances_m = np.abs(result.receiver_x_m - result.source_x_m)

    source_center_s = 1.5 / frequency_hz
    theoretical_times_s = source_center_s + distances_m / velocity_m_s

    # Mesure des pics autour des temps théoriques.
    peak_times_s = []

    for index, expected_s in enumerate(theoretical_times_s):
        signal = result.receiver_signals[:, index]

        mask = np.abs(times_s - expected_s) <= 0.012

        if not np.any(mask):
            raise RuntimeError("Fenêtre de mesure du pic vide.")

        local_times = times_s[mask]
        local_signal = signal[mask]

        if not np.all(np.isfinite(local_signal)):
            raise RuntimeError("Le signal contient des valeurs non finies.")

        peak_index = int(np.argmax(np.abs(local_signal)))
        peak_times_s.append(float(local_times[peak_index]))

    peak_times_s = np.asarray(peak_times_s)

    peak_slope, _ = np.polyfit(distances_m, peak_times_s, 1)
    peak_velocity_m_s = 1.0 / peak_slope

    # Extraction de fenêtres de même longueur, centrées sur les temps
    # théoriques. La corrélation mesure le décalage résiduel entre elles.
    half_window_samples = round(0.020 / dt_s)
    maximum_lag_samples = round(0.006 / dt_s)
    window_length = 2 * half_window_samples + 1

    windows = []

    for index, expected_s in enumerate(theoretical_times_s):
        center_index = int(round(expected_s / dt_s))
        start = center_index - half_window_samples
        stop = center_index + half_window_samples + 1

        signal = result.receiver_signals[:, index]

        if start < 0 or stop > signal.size:
            raise RuntimeError(
                "La fenêtre de corrélation dépasse le signal enregistré."
            )

        window = signal[start:stop].astype(np.float64, copy=True)
        window -= np.mean(window)

        # Réduit l'influence des bords de la fenêtre.
        window *= np.hanning(window_length)

        if not np.all(np.isfinite(window)):
            raise RuntimeError("Une fenêtre contient des valeurs non finies.")

        if np.max(np.abs(window)) <= 0.0:
            raise RuntimeError("Une fenêtre de signal est nulle.")

        windows.append(window)

    residual_lags_s = []
    estimated_intervals_s = []

    for index in range(1, len(windows)):
        previous = windows[index - 1]
        current = windows[index]

        correlation = correlate(
            current,
            previous,
            mode="full",
            method="fft",
        )
        lags = correlation_lags(
            current.size,
            previous.size,
            mode="full",
        )

        allowed = np.abs(lags) <= maximum_lag_samples

        lag_samples = int(
            lags[allowed][np.argmax(correlation[allowed])]
        )

        residual_lag_s = lag_samples * dt_s
        expected_interval_s = (
            theoretical_times_s[index]
            - theoretical_times_s[index - 1]
        )

        residual_lags_s.append(residual_lag_s)
        estimated_intervals_s.append(
            expected_interval_s + residual_lag_s
        )

    residual_lags_s = np.asarray(residual_lags_s)
    estimated_intervals_s = np.asarray(estimated_intervals_s)

    # Reconstruction des retards relatifs à partir des décalages
    # mesurés entre récepteurs voisins.
    relative_times_s = np.concatenate(
        ([0.0], np.cumsum(estimated_intervals_s))
    )

    relative_distances_m = distances_m - distances_m[0]

    correlation_slope, _ = np.polyfit(
        relative_distances_m,
        relative_times_s,
        1,
    )

    if correlation_slope <= 0.0:
        raise RuntimeError(
            "Pente invalide pour l'estimation par corrélation."
        )

    correlation_velocity_m_s = 1.0 / correlation_slope

    # Figure de diagnostic.
    output_path = (
        Path(__file__).resolve().parents[1]
        / "results"
        / "figures"
        / "arrival_crosscorrelation_diagnostic.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    relative_time_axis_ms = (
        np.arange(window_length) - half_window_samples
    ) * dt_s * 1000.0

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(11, 9),
    )

    for index, window in enumerate(windows):
        amplitude = np.max(np.abs(window))
        ax1.plot(
            relative_time_axis_ms,
            window / amplitude,
            linewidth=1.4,
            label=f"x = {result.receiver_x_m[index]:.0f} m",
        )

    ax1.set_title("Signaux alignés autour des temps théoriques")
    ax1.set_xlabel("Temps relatif au centre théorique (ms)")
    ax1.set_ylabel("Amplitude normalisée")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(ncol=3)

    ax2.plot(
        relative_distances_m,
        relative_distances_m / velocity_m_s * 1000.0,
        marker="o",
        linestyle="--",
        label="Retard théorique relatif",
    )
    ax2.plot(
        relative_distances_m,
        relative_times_s * 1000.0,
        marker="s",
        label="Retard estimé par corrélation",
    )

    ax2.set_title("Comparaison des retards relatifs")
    ax2.set_xlabel("Distance relative (m)")
    ax2.set_ylabel("Retard (ms)")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print("=== DIAGNOSTIC PAR CORRÉLATION CROISÉE ===")
    print(f"Vitesse imposée           : {velocity_m_s:.2f} m/s")
    print(f"Vitesse estimée par pics  : {peak_velocity_m_s:.2f} m/s")
    print(
        "Vitesse estimée par corrélation : "
        f"{correlation_velocity_m_s:.2f} m/s"
    )
    print()
    print("Paire de récepteurs | Correction de retard (ms)")

    for index, lag_s in enumerate(residual_lags_s, start=1):
        print(
            f"{result.receiver_x_m[index - 1]:.0f} -> "
            f"{result.receiver_x_m[index]:.0f} m"
            f" | {lag_s * 1000.0:+.3f}"
        )

    print()
    print(f"Figure enregistrée : {output_path}")
    print(
        "Attention : le résultat de corrélation est un diagnostic "
        "complémentaire, pas encore une validation définitive."
    )


if __name__ == "__main__":
    main()
