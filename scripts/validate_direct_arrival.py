"""Validation du temps de propagation avec plusieurs récepteurs."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from seismic.model.velocity_model import VelocityModel
from seismic.wave import simulate_acoustic_wave


def main() -> None:
    # Milieu homogène synthétique.
    dx_m = 2.0
    dz_m = 2.0
    width_m = 240.0
    depth_m = 120.0
    velocity_m_s = 1500.0

    x_m = np.arange(0.0, width_m + dx_m, dx_m)
    z_m = np.arange(0.0, depth_m + dz_m, dz_m)

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

    # Acquisition synthétique.
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

    # Distances source-récepteur et temps théoriques.
    distances_m = np.hypot(
        result.receiver_x_m - result.source_x_m,
        np.full(
            result.receiver_x_m.shape,
            result.receiver_z_m - result.source_z_m,
        ),
    )

    source_center_s = 1.5 / frequency_hz
    theoretical_times_s = source_center_s + distances_m / velocity_m_s

    times_s = result.sample_times_s
    detected_times_s = []

    # Mesurer le pic absolu dans une fenêtre locale autour
    # du temps théorique estimé pour chaque récepteur.
    half_window_s = 0.012

    for index, expected_time_s in enumerate(theoretical_times_s):
        signal = result.receiver_signals[:, index]

        mask = (
            (times_s >= expected_time_s - half_window_s)
            & (times_s <= expected_time_s + half_window_s)
        )

        if not np.any(mask):
            raise RuntimeError(
                f"Aucun échantillon exploitable pour le récepteur "
                f"x={result.receiver_x_m[index]:.1f} m."
            )

        local_times_s = times_s[mask]
        local_signal = signal[mask]

        if not np.all(np.isfinite(local_signal)):
            raise RuntimeError("Le signal contient des valeurs non finies.")

        peak_index = int(np.argmax(np.abs(local_signal)))

        detected_times_s.append(float(local_times_s[peak_index]))

    detected_times_s = np.asarray(detected_times_s, dtype=np.float64)

    if np.any(np.max(np.abs(result.receiver_signals), axis=0) <= 0.0):
        raise RuntimeError("Au moins un récepteur n'a enregistré aucun signal.")

    # Régression linéaire : t_mesuré = intercept + pente * distance.
    slope_s_per_m, intercept_s = np.polyfit(
        distances_m,
        detected_times_s,
        deg=1,
    )

    if not np.isfinite(slope_s_per_m) or slope_s_per_m <= 0.0:
        raise RuntimeError(
            "La pente mesurée n'est pas positive : "
            "impossible d'estimer une vitesse apparente valide."
        )

    apparent_velocity_m_s = 1.0 / slope_s_per_m
    velocity_error_percent = (
        (apparent_velocity_m_s - velocity_m_s)
        / velocity_m_s
        * 100.0
    )

    fitted_times_s = intercept_s + slope_s_per_m * distances_m

    # Figure : signaux enregistrés et relation temps-distance.
    output_path = (
        Path(__file__).resolve().parents[1]
        / "results"
        / "figures"
        / "direct_arrival_validation.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(11, 9),
        gridspec_kw={"height_ratios": [1.5, 1.0]},
    )

    for index, receiver_x in enumerate(result.receiver_x_m):
        signal = result.receiver_signals[:, index]
        peak = float(np.max(np.abs(signal)))

        ax1.plot(
            times_s * 1000.0,
            signal / peak,
            linewidth=1.4,
            label=f"x = {receiver_x:.0f} m",
        )

    ax1.set_title("Signaux enregistrés par les récepteurs")
    ax1.set_xlabel("Temps (ms)")
    ax1.set_ylabel("Amplitude normalisée par trace")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(ncol=3)

    ax2.plot(
        distances_m,
        theoretical_times_s * 1000.0,
        linestyle="--",
        marker="o",
        linewidth=1.7,
        label="Estimation théorique",
    )

    ax2.scatter(
        distances_m,
        detected_times_s * 1000.0,
        s=55,
        marker="s",
        label="Pics mesurés",
        zorder=3,
    )

    ax2.plot(
        distances_m,
        fitted_times_s * 1000.0,
        linestyle="-.",
        linewidth=1.8,
        label=f"Régression : v apparente = {apparent_velocity_m_s:.0f} m/s",
    )

    ax2.set_title("Temps du pic en fonction de la distance")
    ax2.set_xlabel("Distance source-récepteur (m)")
    ax2.set_ylabel("Temps (ms)")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend()

    fig.suptitle(
        "Validation préliminaire de la propagation acoustique",
        fontsize=14,
    )
    fig.tight_layout()

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print("=== VALIDATION MULTI-RÉCEPTEURS ===")
    print(f"Vitesse imposée             : {velocity_m_s:.2f} m/s")
    print(f"Fréquence centrale          : {frequency_hz:.2f} Hz")
    print(f"Pas temporel                : {dt_s:.6f} s")
    print(f"Nombre de CFL               : {result.cfl_number:.4f}")
    print()
    print("Distance (m) | Théorie (ms) | Pic mesuré (ms) | Écart (ms)")

    for distance, expected, detected in zip(
        distances_m,
        theoretical_times_s,
        detected_times_s,
    ):
        print(
            f"{distance:12.3f} | "
            f"{expected * 1000.0:12.3f} | "
            f"{detected * 1000.0:15.3f} | "
            f"{(detected - expected) * 1000.0:+10.3f}"
        )

    print()
    print(f"Pente mesurée                : {slope_s_per_m:.8f} s/m")
    print(f"Vitesse apparente            : {apparent_velocity_m_s:.2f} m/s")
    print(f"Écart relatif sur la vitesse : {velocity_error_percent:+.2f} %")
    print(f"Figure enregistrée           : {output_path}")
    print()
    print(
        "INTERPRÉTATION : cette expérience évalue la relation "
        "temps-distance. Elle ne valide pas à elle seule "
        "toutes les propriétés du solveur."
    )


if __name__ == "__main__":
    main()
