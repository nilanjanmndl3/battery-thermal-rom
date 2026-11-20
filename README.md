# LFP prismatic battery pack liquid-cooling ROM

## Contents
- `lfp_serpentine_rom.py`: coupled battery-core / cell-surface / cold-plate / coolant ROM.
- Running the script produces `rom_results.csv`.

## Install
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install numpy scipy pandas matplotlib
```

## Run
```bash
python lfp_serpentine_rom.py
```

## Model assumptions and limitations
1. Cells are represented by two thermal nodes (core and surface).
2. Each cell is coupled to one cold-plate segment; the coolant segments are connected in series.
3. Coolant flow is prescribed and treated as constant. The hydraulic calculation is a separate estimate.
4. The example electrical model uses a simple SOC-to-OCV curve, constant entropic coefficient, and Arrhenius temperature dependence of resistance.
5. The default parameters are illustrative, not specific to a commercial LFP cell.
6. The cold-plate model is a 1-D network. It does not resolve 3-D spreading, U-bend secondary flow, local maldistribution, or conjugate details.

## Important calibration sequence
1. Measure or obtain cell mass, heat capacity, core/surface conductance, cell resistance and OCV-SOC-temperature maps.
2. Obtain cold-plate dimensions, material properties, coolant mixture properties, mass flow and contact resistance.
3. Run 3-D conjugate CFD for representative flow rates and heat loads.
4. Export area-averaged local heat-transfer coefficients (or segment-wise UA), coolant pressure drop, and spatial temperature data.
5. Calibrate the ROM using multiple operating points, not one transient alone.
6. Validate on separate current profiles and flow rates. Report RMSE, MAE, bias, maximum error and computational speed-up.

## CFD data mapping
Map the serpentine centreline into N ordered channel segments. For each segment, extract:
- segment-averaged coolant bulk temperature,
- plate-side wall temperature,
- cell/plate interface temperature,
- local heat rate,
- local effective heat-transfer coefficient,
- pressure drop across the channel.

The current `calibrate_htc_scale` function fits one global multiplier on coolant-side UA from temperature observations. For publication-quality calibration, extend it to a small number of spatial HTC zones and jointly fit thermal contact conductance where identifiable.

## Sign conventions
- Positive current means discharge.
- SOC decreases during positive current.
- Heat generation uses `I^2 R - I*T*dUocv/dT`.
- Temperatures are in kelvin internally and converted to Celsius only for output.

## Suggested validation matrix
| Case | C-rate | Coolant flow | Inlet temperature | Purpose |
|---|---:|---:|---:|---|
| A | 0.5C | low | 20°C | low-load baseline |
| B | 1C | nominal | 20°C | nominal validation |
| C | 2C | nominal | 20°C | high heat load |
| D | 1C | high | 20°C | flow sensitivity |
| E | 1C | nominal | 30°C | inlet-temperature sensitivity |

Do not use a fitted case as an independent validation case.
