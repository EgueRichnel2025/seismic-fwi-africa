"""Étude de convergence de la propagation en milieu homogène."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import correlate, correlation_lags

from seismic.model.velocity_model import VelocityModel
from seismic.wave import simulate_acoustic_wave


TRUE_VELOCITY_M_S = 1500.0
SOURCE_X_M = 60.0
SOURCE_Z_M = 50.0
RECEIVER_X_M = np.array(
    [80.0, 100.0, 120.0, 140.0, 160.0],
    dtype=np.float64,
)
RECEIVER_Z_M = 50.0
FREQUENCY_HZ = 25.0
DURATION_S = 0.16


def estimate_velocity(
    dx_m: float,
    dt_s: float,
) -> tuple[float, np.ndarray]:
    """Simule le milieu homogène et estime la vitesse apparente."""

    dz_m = dx_m
    width_m = 240.0
    depth_m = 120.0

    x_m = np.arange(0.0, width_m + 0.5 * dx_m, dx_m)
    z_m = np.arange(0.0, depth_m + 0.5 * dz_m, dz_m)

    velocity = np.full(
        (z_m.size, x_m.size),
        TRUE_VELOCITY_M_S,
        dtype=np.float64,
    )

    model = VelocityModel(
        x_m=x_m,
        z_m=z_m,
        velocity_m_s=velocity,
        anomaly_mask=np.zeros_like(velocity, dtype=bool),
    )

    result = simulate_acoustic_wave(
        model,
        duration_s=DURATION_S,
        dt_s=dt_s,
        source_x_m=SOURCE_X_M,
        source_z_m=SOURCE_Z_M,
        source_frequency_hz=FREQUENCY_HZ,
        receiver_x_m=RECEIVER_X_M,
        receiver_z_m=RECEIVER_Z_M,
        sponge_width_m=40.0,
        sponge_strength_s_inv=80.0,
        snapshot_stride=max(1, round(0.01 / dt_s)),
    )

    distances_m = np.abs(result.receiver_x_m - SOURCE_X_M)
    times_s = result.sample_times_s

    source_center_s = 1.5 / FREQUENCY_HZ
    theoretical_times_s = (
        source_center_s + distances_m / TRUE_VELOCITY_M_S
    )

    half_window_s = 0.020
    half_window_samples = round(half_window_s / dt_s)
    maximum_lag_samples = round(0.006 / dt_s)
    window_length = 2 * half_window_samples + 1

    windows = []

    for receiver_index, expected_time_s in enumerate(
        theoretical_times_s
    ):
        center_index = int(round(expected_time_s / dt_s))

        start = center_index - half_window_samples
        stop = center_index + half_window_samples + 1

        signal = result.receiver_signals[:, receiver_index]

        if start < 0 or stop > signal.size:
            raise RuntimeError(
                "La fenêtre de corrélation dépasse le signal disponible."
            )

        window = signal[start:stop].copy()
        window -= np.mean(window)
        window *= np.hanning(window_length)

        if not np.all(np.isfinite(window)):
            raise RuntimeError("Une fenêtre contient des valeurs non finies.")

        if np.max(np.abs(window)) <= 0.0:
            raise RuntimeError("Une fenêtre de signal est nulle.")

        windows.append(window)

    residual_lags_s = []

    for index in range(1, len(windows)):
        previous = windows[index - 1]
        current = windows[index]

        cross_correlation = correlate(
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

        allowed_indices = np.flatnonzero(
            np.abs(lags) <= maximum_lag_samples
        )

        peak_index = int(
            allowed_indices[
                np.argmax(cross_correlation[allowed_indices])
            ]
        )

        lag_samples = float(lags[peak_index])

        # Raffinement sous-échantillon par interpolation parabolique
        # autour du maximum de corrélation.
        if 0 < peak_index < cross_correlation.size - 1:
            y_minus = cross_correlation[peak_index - 1]
            y_zero = cross_correlation[peak_index]
            y_plus = cross_correlation[peak_index + 1]

            denominator = y_minus - 2.0 * y_zero + y_plus

            if denominator < 0.0:
                correction = (
                    0.5 * (y_minus - y_plus) / denominator
                )
                lag_samples += float(np.clip(correction, -1.0, 1.0))

        residual_lags_s.append(lag_samples * dt_s)

    residual_lags_s = np.asarray(residual_lags_s)

    theoretical_intervals_s = np.diff(theoretical_times_s)
    estimated_intervals_s = (
        theoretical_intervals_s + residual_lags_s
    )

    relative_distances_m = distances_m - distances_m[0]
    relative_times_s = np.concatenate(
        ([0.0], np.cumsum(estimated_intervals_s))
    )

    slope_s_per_m, _ = np.polyfit(
        relative_distances_m,
        relative_times_s,
        deg=1,
    )

    if not np.isfinite(slope_s_per_m) or slope_s_per_m <= 0.0:
        raise RuntimeError(
            "La pente mesurée ne permet pas une vitesse valide."
        )

    apparent_velocity_m_s = 1.0 / slope_s_per_m

    return apparent_velocity_m_s, residual_lags_s


def main() -> None:
    configurations = [
        ("A : dx=2 m, dt=0.4 ms", 2.0, 4.0e-4),
        ("B : dx=2 m, dt=0.2 ms", 2.0, 2.0e-4),
        ("C : dx=1 m, dt=0.2 ms", 1.0, 2.0e-4),
    ]

    labels = []
    velocities_m_s = []

    print("=== ÉTUDE DE CONVERGENCE NUMÉRIQUE ===")
    print(f"Vitesse imposée : {TRUE_VELOCITY_M_S:.2f} m/s")
    print()

    for label, dx_m, dt_s in configurations:
        print(f"Configuration : {label}")

        apparent_velocity, residual_lags = estimate_velocity(
            dx_m=dx_m,
            dt_s=dt_s,
        )

        relative_error_percent = (
            (apparent_velocity - TRUE_VELOCITY_M_S)
            / TRUE_VELOCITY_M_S
            * 100.0
        )

        print(f"  Pas spatial             : {dx_m:.2f} m")
        print(f"  Pas temporel            : {dt_s * 1000.0:.3f} ms")
        print(f"  Vitesse apparente       : {apparent_velocity:.2f} m/s")
        print(f"  Écart relatif           : {relative_error_percent:+.3f} %")
        print(
            "  Retards résiduels       : "
            + ", ".join(
                f"{lag * 1000.0:+.3f} ms"
                for lag in residual_lags
            )
        )
        print()

        labels.append(label)
        velocities_m_s.append(apparent_velocity)

    output_path = (
        Path(__file__).resolve().parents[1]
        / "results"
        / "figures"
        / "grid_convergence.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5.8))

    ax.plot(
        labels,
        velocities_m_s,
        marker="o",
        linewidth=2.0,
        markersize=7,
        label="Vitesse apparente estimée",
    )

    ax.axhline(
        TRUE_VELOCITY_M_S,
        linestyle="--",
        linewidth=1.8,
        label="Vitesse imposée : 1 500 m/s",
    )

    for index, value in enumerate(velocities_m_s):
        ax.annotate(
            f"{value:.1f} m/s",
            (index, value),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
        )

    ax.set_title("Convergence de l'estimation de la vitesse")
    ax.set_ylabel("Vitesse apparente (m/s)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    ax.tick_params(axis="x", labelrotation=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Figure enregistrée : {output_path}")
    print(
        "Cette étude compare la stabilité de l'estimation. "
        "Elle ne constitue pas, à elle seule, une validation "
        "complète du solveur."
    )


if __name__ == "__main__":
    main()
