# Configuration reference

Regional workflows load YAML into `SuetesConfig`. Missing sections and keys use
dataclass defaults; unknown keys raise an error. Preprocessing and simulation
call the same resolver so geometry, dates, cache names, and stores agree.

## Sections

| Section | Important fields | Purpose |
|---|---|---|
| `domain` | `name`, `lat_c`, `lon_c`, `nx`, `ny`, `nz`, `dx`, `dy`, `dz` | Projection center and logical grid |
| `time` | `year`, `month`, `days`, `sim_hours` | Input dates and integration extent |
| `vertical` | `kappa`, `scale_s`, `n` | Simplified or stretched SLEVE mapping |
| `core` | `type`, `dt`, `nu_h`, `ns`, `alpha`, `precision` | Integration and numerical controls |
| `physics` | process switches and values | Regional parameterizations |
| `constants` | `g`, `Rd`, `cp`, `cvd`, `p0`, `epsilon` | Thermodynamic constants |
| `io` | paths and preprocessing settings | Data movement and boundary construction |
| `render` | variables, levels, hours, points, extents | Plot selection |

`core.type` accepts `sisl` or `split-explicit`. Loading `precision` configures
JAX x64 globally. For the vertical mapping, `kappa > 1` selects stretched
SLEVE; otherwise the resolver selects simplified SLEVE.

## Example fragment

```yaml
domain:
  name: example
  lat_c: 47.5
  lon_c: -59.0
  nx: 120
  ny: 100
  nz: 48
  dx: 2500.0
  dy: 2500.0
  dz: 400.0

time:
  year: "2025"
  month: "02"
  days: ["18", "19"]
  sim_hours: 24

vertical:
  kappa: 1.0
  scale_s: 10000.0
  n: 1.0

core:
  type: split-explicit
  dt: 10.0
  ns: 3
  precision: float32
```

Use a repository configuration as the basis for a real run; this fragment
intentionally omits physics, I/O, and rendering choices.

## Environment overrides

The resolver recognizes `SIM_HOURS`, `KAPPA`, `START_DAY`, `CORE_TYPE`, `DT`,
`NU_H`, `NS`, `ALPHA`, `RH_CRIT`, `RAD_COARSE`, `RAD_EVERY_H`,
`DOWNLOAD_WORKERS`, and process switches such as `NO_RAD`, `NO_NUDGE`,
`NO_MICRO`, `NO_CONV`, `NO_DRAG`, `NO_DIFFUSION`, `NO_SGS`, `NO_GWD`,
`USE_SGS`, and `USE_GWD`.

Overrides take precedence over YAML. Prefer versioned YAML for scientific
settings and record any overrides in provenance. The `NO_*` flags follow the
current resolver implementation; inspect resolved settings rather than
inferring the effective choice from the name alone.

## Input and cache identity

ERA5 download names encode geographic domain, date, buffer, and pressure preset,
allowing numerical grids with different resolution to reuse a raw subset.
Processed-store names additionally encode model domain, resolution, number of
states, and vertical-coordinate tag.

See the generated [configuration API](api/shared.md#suetes.shared.config) for
the current dataclasses and helper signatures.
