"""
Physics-based reduced-order model for an LFP prismatic battery pack
with a liquid-cooled serpentine cold plate.

Model structure:
  - one core and one surface thermal node per cell
  - one cold-plate thermal node per cell/channel segment
  - one coolant thermal node per channel segment
  - serial coolant advection along the serpentine flow path
  - SOC-dependent electrical heat generation
  - optional entropic (reversible) heat
  - lateral thermal conduction between neighbouring cells/plate segments
  - estimated channel pressure drop and pumping power

This is a parameterized research starting point, not a validated design tool.
Replace illustrative parameters and calibrate against experiments/3-D CFD.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Optional
import json
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares


@dataclass
class BatteryParameters:
    n_cells: int = 12
    capacity_ah: float = 100.0
    cell_mass_kg: float = 2.4
    cell_cp_j_kgk: float = 950.0
    core_capacity_fraction: float = 0.75
    # Effective core-to-surface conductance for each cell
    g_core_surface_w_k: float = 8.0
    # Cell-surface-to-cold-plate conductance per cell, including contact resistance
    g_surface_plate_w_k: float = 12.0
    # Lateral conductance between adjacent cell surfaces
    g_cell_lateral_w_k: float = 0.8
    # Cell-to-ambient loss (set to zero for an insulated pack)
    g_cell_ambient_w_k: float = 0.15
    ambient_temperature_k: float = 298.15

    # Illustrative equivalent-circuit parameters. Replace with measured maps.
    r0_ohm_ref: float = 0.0018
    r0_activation_energy_j_mol: float = 12000.0
    reference_temperature_k: float = 298.15
    ocv_soc: tuple = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
                      0.6, 0.7, 0.8, 0.9, 1.0)
    ocv_v: tuple = (2.80, 3.15, 3.22, 3.25, 3.27, 3.29,
                    3.30, 3.31, 3.32, 3.33, 3.35)
    # dUocv/dT in V/K; illustrative constant. Prefer measured SOC map.
    entropic_coefficient_v_k: float = 0.00005


@dataclass
class ColdPlateParameters:
    # Serpentine channel geometry
    channel_width_m: float = 0.006
    channel_height_m: float = 0.002
    channel_total_length_m: float = 1.8
    number_of_passes: int = 6
    channel_roughness_m: float = 1.5e-6
    minor_loss_coefficient: float = 8.0

    # Cold plate material and discretization
    plate_length_m: float = 0.30
    plate_width_m: float = 0.18
    plate_thickness_m: float = 0.004
    plate_density_kg_m3: float = 2700.0
    plate_cp_j_kgk: float = 900.0
    plate_conductivity_w_mk: float = 170.0

    # Effective coolant-side heat transfer coefficient. Calibrate with CFD.
    h_coolant_w_m2k: float = 1200.0
    # Effective wetted perimeter; for a rectangular duct this is 2(w+h).
    coolant_inlet_temperature_k: float = 293.15
    coolant_mass_flow_kg_s: float = 0.04

    # Water-glycol-like placeholder properties; use actual mixture data.
    coolant_density_kg_m3: float = 1030.0
    coolant_cp_j_kgk: float = 3600.0
    coolant_viscosity_pa_s: float = 0.0025
    coolant_conductivity_w_mk: float = 0.40


@dataclass
class SimulationSettings:
    t_start_s: float = 0.0
    t_end_s: float = 3600.0
    output_points: int = 361
    initial_cell_temperature_k: float = 298.15
    initial_plate_temperature_k: float = 298.15
    initial_coolant_temperature_k: float = 293.15
    initial_soc: float = 0.95
    solver_method: str = "BDF"
    rtol: float = 1e-6
    atol: float = 1e-8
    max_step_s: Optional[float] = 10.0


class LFPSerpentineROM:
    """Lumped-parameter electro-thermal ROM with a serial coolant path."""

    R_GAS = 8.314462618

    def __init__(
        self,
        battery: BatteryParameters = BatteryParameters(),
        plate: ColdPlateParameters = ColdPlateParameters(),
        settings: SimulationSettings = SimulationSettings(),
    ):
        self.b = battery
        self.p = plate
        self.s = settings
        self.n = battery.n_cells
        self._validate_parameters()
        self._precompute_thermal_network()

    def _validate_parameters(self):
        if self.b.n_cells < 2:
            raise ValueError("n_cells must be at least 2.")
        if not (0.0 < self.b.core_capacity_fraction < 1.0):
            raise ValueError("core_capacity_fraction must lie between 0 and 1.")
        if self.p.channel_width_m <= 0 or self.p.channel_height_m <= 0:
            raise ValueError("Channel dimensions must be positive.")
        if self.p.coolant_mass_flow_kg_s <= 0:
            raise ValueError("Coolant mass flow must be positive.")
        if self.p.channel_total_length_m <= 0:
            raise ValueError("Channel length must be positive.")
        if self.p.coolant_density_kg_m3 <= 0:
            raise ValueError("Coolant density must be positive.")

    def _precompute_thermal_network(self):
        n, b, p = self.n, self.b, self.p
        c_cell_total = b.cell_mass_kg * b.cell_cp_j_kgk
        self.c_core = np.full(n, c_cell_total * b.core_capacity_fraction)
        self.c_surface = np.full(n, c_cell_total * (1-b.core_capacity_fraction))

        dx = p.channel_total_length_m / n
        self.dx = dx
        channel_area = p.channel_width_m * p.channel_height_m
        self.channel_area = channel_area
        self.hydraulic_diameter = (
            2*p.channel_width_m*p.channel_height_m /
            (p.channel_width_m+p.channel_height_m)
        )
        self.channel_volume_segment = channel_area * dx
        self.c_fluid = np.full(
            n,
            p.coolant_density_kg_m3 * self.channel_volume_segment *
            p.coolant_cp_j_kgk
        )

        # Allocate plate thermal mass to channel-associated segments.
        plate_volume = p.plate_length_m*p.plate_width_m*p.plate_thickness_m
        self.c_plate = np.full(
            n, p.plate_density_kg_m3*plate_volume*p.plate_cp_j_kgk/n
        )

        wetted_perimeter = 2*(p.channel_width_m+p.channel_height_m)
        self.ua_pf = np.full(n, p.h_coolant_w_m2k*wetted_perimeter*dx)

        # Effective lateral conductance through the plate in the flow direction.
        cross_section = p.plate_width_m*p.plate_thickness_m
        self.g_plate_lateral = (
            p.plate_conductivity_w_mk*cross_section/max(dx, 1e-12)
        )

    def _ocv(self, soc):
        return float(np.interp(
            np.clip(soc, 0.0, 1.0),
            np.asarray(self.b.ocv_soc),
            np.asarray(self.b.ocv_v)
        ))

    def _resistance(self, temperature_k):
        b = self.b
        exponent = b.r0_activation_energy_j_mol/self.R_GAS * (
            1.0/temperature_k - 1.0/b.reference_temperature_k
        )
        return b.r0_ohm_ref*np.exp(np.clip(exponent, -5.0, 5.0))

    def _current(self, t, current_profile):
        """Positive current means discharge. Profile may be scalar or callable."""
        value = current_profile(t) if callable(current_profile) else current_profile
        return float(value)

    def _rhs(self, t, y, current_profile, h_scale=1.0):
        n = self.n
        b, p = self.b, self.p

        # State vector: [Tcore(n), Tsurface(n), Tplate(n), Tfluid(n), SOC]
        tc = y[0:n]
        ts = y[n:2*n]
        tp = y[2*n:3*n]
        tf = y[3*n:4*n]
        soc = float(y[4*n])

        d_tc = np.zeros(n)
        d_ts = np.zeros(n)
        d_tp = np.zeros(n)
        d_tf = np.zeros(n)

        current = self._current(t, current_profile)
        cell_current = current  # Pack current through each series-connected cell
        soc_rate = -cell_current/(3600.0*b.capacity_ah)

        # Cell electrical heat generation: I^2 R - I*T*dUocv/dT.
        # Assumes identical series cells and a common pack current.
        for i in range(n):
            r_i = self._resistance(max(tc[i], 200.0))
            q_irrev = cell_current**2*r_i
            q_reversible = (
                -cell_current*tc[i]*b.entropic_coefficient_v_k
            )
            q_gen = q_irrev + q_reversible

            q_core_surface = b.g_core_surface_w_k*(tc[i]-ts[i])
            q_surface_plate = b.g_surface_plate_w_k*(ts[i]-tp[i])

            q_lateral_cell = 0.0
            if i > 0:
                q_lateral_cell += b.g_cell_lateral_w_k*(ts[i-1]-ts[i])
            if i < n-1:
                q_lateral_cell += b.g_cell_lateral_w_k*(ts[i+1]-ts[i])

            q_ambient = b.g_cell_ambient_w_k*(
                b.ambient_temperature_k-ts[i]
            )

            d_tc[i] = (q_gen-q_core_surface)/self.c_core[i]
            d_ts[i] = (
                q_core_surface-q_surface_plate+q_lateral_cell+q_ambient
            )/self.c_surface[i]

            q_plate_from_cell = q_surface_plate

            # Plate conduction to adjacent segments.
            q_plate_lateral = 0.0
            if i > 0:
                q_plate_lateral += self.g_plate_lateral*(tp[i-1]-tp[i])
            if i < n-1:
                q_plate_lateral += self.g_plate_lateral*(tp[i+1]-tp[i])

            ua = self.ua_pf[i]*h_scale
            q_plate_to_fluid = ua*(tp[i]-tf[i])
            d_tp[i] = (
                q_plate_from_cell-q_plate_to_fluid+q_plate_lateral
            )/self.c_plate[i]

            # Serial advection: coolant enters segment 0 at inlet temperature.
            t_upstream = (
                p.coolant_inlet_temperature_k if i == 0 else tf[i-1]
            )
            q_advection = (
                p.coolant_mass_flow_kg_s*p.coolant_cp_j_kgk*
                (t_upstream-tf[i])
            )
            d_tf[i] = (q_advection+q_plate_to_fluid)/self.c_fluid[i]

        # Stop SOC integration at the physical limits.
        if (soc <= 0.0 and soc_rate < 0.0) or (soc >= 1.0 and soc_rate > 0.0):
            soc_rate = 0.0

        return np.concatenate([d_tc, d_ts, d_tp, d_tf, [soc_rate]])

    def simulate(self, current_profile=100.0, h_scale=1.0):
        """Run transient simulation. current_profile is positive on discharge."""
        n, s = self.n, self.s
        y0 = np.concatenate([
            np.full(n, s.initial_cell_temperature_k),  # core
            np.full(n, s.initial_cell_temperature_k),  # surface
            np.full(n, s.initial_plate_temperature_k),
            np.full(n, s.initial_coolant_temperature_k),
            [s.initial_soc]
        ])
        t_eval = np.linspace(s.t_start_s, s.t_end_s, s.output_points)
        kwargs = dict(
            fun=lambda t, y: self._rhs(t, y, current_profile, h_scale),
            t_span=(s.t_start_s, s.t_end_s),
            y0=y0,
            t_eval=t_eval,
            method=s.solver_method,
            rtol=s.rtol,
            atol=s.atol
        )
        if s.max_step_s is not None:
            kwargs["max_step"] = s.max_step_s
        sol = solve_ivp(**kwargs)
        if not sol.success:
            raise RuntimeError(f"ODE integration failed: {sol.message}")

        result = {
            "time_s": sol.t,
            "core_K": sol.y[0:n, :].T,
            "surface_K": sol.y[n:2*n, :].T,
            "plate_K": sol.y[2*n:3*n, :].T,
            "coolant_K": sol.y[3*n:4*n, :].T,
            "soc": np.clip(sol.y[4*n, :], 0.0, 1.0),
            "current_A": np.array([self._current(t, current_profile) for t in sol.t])
        }
        result["max_cell_K"] = np.max(result["core_K"], axis=1)
        result["min_cell_K"] = np.min(result["core_K"], axis=1)
        result["cell_spread_K"] = (
            result["max_cell_K"]-result["min_cell_K"]
        )
        result["outlet_coolant_K"] = result["coolant_K"][:, -1]
        return result

    def hydraulic_estimate(self):
        """Single-channel pressure-drop estimate; laminar Darcy or Blasius turbulent."""
        p = self.p
        q_vol = p.coolant_mass_flow_kg_s/p.coolant_density_kg_m3
        velocity = q_vol/self.channel_area
        re = (
            p.coolant_density_kg_m3*velocity*self.hydraulic_diameter /
            p.coolant_viscosity_pa_s
        )
        if re < 2300:
            f = 64.0/max(re, 1e-12)
            regime = "laminar"
        else:
            # Smooth-pipe Blasius approximation; use Colebrook/Churchill for
            # rough/turbulent channels in a final design model.
            f = 0.3164/(re**0.25)
            regime = "turbulent approximation"
        dp_major = (
            f*(p.channel_total_length_m/self.hydraulic_diameter)*
            0.5*p.coolant_density_kg_m3*velocity**2
        )
        dp_minor = (
            p.minor_loss_coefficient*
            0.5*p.coolant_density_kg_m3*velocity**2
        )
        dp = dp_major+dp_minor
        hydraulic_power = dp*q_vol
        pump_electrical_power = hydraulic_power/0.5  # assumed pump efficiency
        return {
            "Re": re,
            "velocity_m_s": velocity,
            "friction_factor": f,
            "regime": regime,
            "pressure_drop_Pa": dp,
            "hydraulic_power_W": hydraulic_power,
            "pump_electrical_power_W_at_50pct_efficiency": pump_electrical_power
        }


def compare_rom_to_cfd(rom_values, cfd_values):
    """Compare matching arrays (same locations/times); returns standard metrics."""
    rom = np.asarray(rom_values, dtype=float).ravel()
    cfd = np.asarray(cfd_values, dtype=float).ravel()
    if rom.shape != cfd.shape:
        raise ValueError("ROM and CFD arrays must have matching shapes.")
    error = rom-cfd
    return {
        "RMSE_K": float(np.sqrt(np.mean(error**2))),
        "MAE_K": float(np.mean(np.abs(error))),
        "maximum_absolute_error_K": float(np.max(np.abs(error))),
        "bias_K": float(np.mean(error)),
        "R2": float(
            1.0-np.sum(error**2)/
            max(np.sum((cfd-np.mean(cfd))**2), 1e-15)
        )
    }


def calibrate_htc_scale(model, cfd_or_experimental_temperatures_K,
                        measurement_times_s, measurement_cell_indices,
                        current_profile=100.0, initial_h_scale=1.0):
    """
    Fit one global multiplier on coolant-side UA using measured/CFD cell-core
    temperatures. For spatially resolved CFD data, extend this to segment-wise
    multipliers or a low-dimensional basis for h(x).
    """
    target = np.asarray(cfd_or_experimental_temperatures_K, dtype=float)
    times = np.asarray(measurement_times_s, dtype=float)
    cells = np.asarray(measurement_cell_indices, dtype=int)
    if not (target.shape == times.shape == cells.shape):
        raise ValueError("Target temperatures, times and cell indices must match.")

    original_end = model.s.t_end_s
    original_points = model.s.output_points
    model.s.t_end_s = max(original_end, float(np.max(times)))
    model.s.output_points = max(original_points, 2)

    def residual(log_scale):
        scale = float(np.exp(log_scale[0]))
        sim = model.simulate(current_profile=current_profile, h_scale=scale)
        pred = np.array([
            np.interp(t, sim["time_s"], sim["core_K"][:, c])
            for t, c in zip(times, cells)
        ])
        return pred-target

    fit = least_squares(
        residual, x0=[np.log(initial_h_scale)],
        bounds=([np.log(0.05)], [np.log(20.0)])
    )
    model.s.t_end_s = original_end
    model.s.output_points = original_points
    return {
        "h_scale": float(np.exp(fit.x[0])),
        "cost": float(fit.cost),
        "success": bool(fit.success),
        "message": fit.message
    }


def save_results(result, filename="rom_results.csv"):
    """Save cell-wise temperature histories and pack-level metrics."""
    n = result["core_K"].shape[1]
    data = {
        "time_s": result["time_s"],
        "SOC": result["soc"],
        "current_A": result["current_A"],
        "max_cell_core_C": result["max_cell_K"]-273.15,
        "min_cell_core_C": result["min_cell_K"]-273.15,
        "cell_spread_K": result["cell_spread_K"],
        "coolant_outlet_C": result["outlet_coolant_K"]-273.15,
    }
    for i in range(n):
        data[f"cell_{i+1}_core_C"] = result["core_K"][:, i]-273.15
        data[f"cell_{i+1}_surface_C"] = result["surface_K"][:, i]-273.15
        data[f"plate_segment_{i+1}_C"] = result["plate_K"][:, i]-273.15
        data[f"coolant_segment_{i+1}_C"] = result["coolant_K"][:, i]-273.15
    pd.DataFrame(data).to_csv(filename, index=False)


def main():
    battery = BatteryParameters(n_cells=12)
    plate = ColdPlateParameters(
        coolant_density_kg_m3=1030.0,
        coolant_mass_flow_kg_s=0.04,
        h_coolant_w_m2k=1200.0
    )
    settings = SimulationSettings(t_end_s=3600.0, output_points=361)

    model = LFPSerpentineROM(battery, plate, settings)

    # Example: 1C discharge for a 100 Ah series string.
    result = model.simulate(current_profile=100.0)

    hydraulic = model.hydraulic_estimate()
    final_core_c = result["core_K"][-1]-273.15
    final_surface_c = result["surface_K"][-1]-273.15

    print("\n--- ROM summary ---")
    print(f"Final SOC: {result['soc'][-1]:.4f}")
    print(f"Maximum core temperature: {final_core_c.max():.2f} °C")
    print(f"Minimum core temperature: {final_core_c.min():.2f} °C")
    print(f"Cell core temperature spread: {np.ptp(final_core_c):.3f} K")
    print(f"Coolant outlet temperature: {result['outlet_coolant_K'][-1]-273.15:.2f} °C")
    print("\n--- Hydraulic estimate ---")
    print(json.dumps(hydraulic, indent=2))

    save_results(result, "rom_results.csv")
    print("\nSaved rom_results.csv")


if __name__ == "__main__":
    main()
