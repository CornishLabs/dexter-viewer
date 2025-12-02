# %% Imports
import xml.etree.ElementTree as ET
from itertools import islice
from dataclasses import dataclass
from typing import Optional

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

# %% Parse file
xml_path = "example-sequence-2.xml"

parser = ET.XMLParser(encoding="cp1252")  # try "latin-1" if needed
tree = ET.parse(xml_path, parser=parser)
root = tree.getroot()

print("=== Step 1: Parsed XML ===")
print("Root tag:", root.tag)
print("Root attributes:", root.attrib)
print("Number of direct children:", len(root))

# %% Explore structure
def outline(elem, depth=0, max_depth=3, max_children=20):
    indent = "  " * depth
    print(f"{indent}<{elem.tag}> attrs={elem.attrib} text={repr((elem.text or '').strip())[:60]}")
    if depth >= max_depth:
        return
    for child in list(elem)[:max_children]:
        outline(child, depth + 1, max_depth, max_children)
    if len(elem) > max_children:
        print(f"{indent}  ... ({len(elem) - max_children} more children)")

print("\n=== Step 2: Outline (peek) ===")
outline(root, max_depth=6, max_children=15)

# %% Look at names we've seen
NS = {"lv": "http://www.ni.com/LVData"}

def txt(e: Optional[ET.Element]) -> str:
    return (e.text or "").strip() if e is not None else ""

def local_tag(tag: str) -> str:
    # "{namespace}Tag" -> "Tag"
    return tag.split("}", 1)[1] if "}" in tag else tag

def find_named_arrays(root: ET.Element, target_name: str) -> list[ET.Element]:
    """Return all <Array> elements whose child <Name> equals target_name."""
    out = []
    for arr in root.findall(".//lv:Array", NS):
        name_el = arr.find("lv:Name", NS)
        if name_el is not None and txt(name_el) == target_name:
            out.append(arr)
    return out

def get_named_array(root: ET.Element, target_name: str) -> ET.Element:
    matches = find_named_arrays(root, target_name)
    if not matches:
        raise KeyError(f"Could not find an <lv:Array> named: {target_name!r}")
    if len(matches) > 1:
        print(f"Warning: found {len(matches)} arrays named {target_name!r}; using the first.")
    return matches[0]

def array_dims(arr: ET.Element) -> list[int]:
    return [int(txt(d)) for d in arr.findall("lv:Dimsize", NS)]

def preview_bool_array(arr: ET.Element, n=12) -> list[int]:
    vals = [int(txt(v.find("lv:Val", NS))) for v in islice(arr.findall("lv:Boolean", NS), n)]
    return vals

print("\n=== Step 3: Locate & summarize expected arrays ===")
for key in [
    "Sequence header top",
    "Fast digital channels",
    "Fast digital names",
    "Slow digital channels",
    "Slow digital names",
    "Fast analogue array",
    "Fast analogue names",
    "Slow analogue array",
    "Slow analogue names",
]:
    matches = find_named_arrays(root, key)
    print(f"\n{key}: found {len(matches)}")
    for i, arr in enumerate(matches):
        dims = array_dims(arr)
        print(f"  [{i}] dims={dims}")
        if arr.find("lv:Boolean", NS) is not None:
            print("     first vals:", preview_bool_array(arr, n=16))

# %% Define Dataclasses
@dataclass(frozen=True)
class ChannelName:
    hardware_id: str
    human_name: str

@dataclass(frozen=True)
class ChannelSet:
    label: str
    channels: list[ChannelName]

    @property
    def count(self) -> int:
        return len(self.channels)

    @property
    def hardware_ids(self) -> list[str]:
        return [c.hardware_id for c in self.channels]

    @property
    def human_names(self) -> list[str]:
        return [c.human_name for c in self.channels]

    @property
    def hardware_to_index(self) -> dict[str, int]:
        return {c.hardware_id: i for i, c in enumerate(self.channels)}

    @property
    def human_to_index(self) -> dict[str, int]:
        return {c.human_name: i for i, c in enumerate(self.channels)}

    @property
    def pairs(self) -> list[tuple[str, str]]:
        """Convenience: [(hardware_id, human_name), ...]"""
        return [(c.hardware_id, c.human_name) for c in self.channels]

@dataclass(frozen=True)
class DigitalMatrices:
    fast: np.ndarray  # (channels, steps) bool
    slow: np.ndarray  # (channels, steps) bool

@dataclass(frozen=True)
class AnalogueMatrices:
    fast_voltage: np.ndarray  # (channels, steps) float
    fast_ramp: np.ndarray     # (channels, steps) bool
    slow_voltage: np.ndarray  # (channels, steps) float
    slow_ramp: np.ndarray     # (channels, steps) bool

@dataclass(frozen=True)
class StepHeader:
    idx: int
    event_name: str
    step_name: str
    dt_value: float
    dt_unit: str
    dt_seconds: Optional[float]
    event_id: Optional[int]
    hide_event_steps: Optional[int]
    populate_multirun: Optional[int]
    skip_step: Optional[int]

# %% Parse Channel names (hardware + human), and print a summary.
def parse_channel_pairs_from_cluster_array(names_arr: ET.Element) -> list[ChannelName]:
    """
    Name arrays are <Array> of <Cluster>, each cluster containing:
      - <String><Name>Hardware ID</Name><Val>FDO 0</Val></String>
      - <String><Name>Name</Name><Val>977 P0</Val></String>
    """
    out: list[ChannelName] = []
    clusters = names_arr.findall("lv:Cluster", NS)

    for idx, cl in enumerate(clusters):
        hardware_id = None
        human_name = None

        for s in cl.findall(".//lv:String", NS):
            key = txt(s.find("lv:Name", NS))
            val = txt(s.find("lv:Val", NS))
            if key == "Hardware ID":
                hardware_id = val
            elif key == "Name":
                human_name = val

        if hardware_id is None:
            hardware_id = f"(missing_hw_{idx})"
        if human_name is None:
            first_val = None
            for s in cl.findall(".//lv:String", NS):
                v = txt(s.find("lv:Val", NS))
                if v:
                    first_val = v
                    break
            human_name = first_val if first_val is not None else f"(missing_name_{idx})"

        out.append(ChannelName(hardware_id=hardware_id, human_name=human_name))

    return out

def summarize_channelset(cs: ChannelSet, max_preview: int = 6):
    print(f"\n--- {cs.label} ---")
    print("Count:", cs.count)
    print("First pairs (hardware_id -> human_name):")
    for hw, name in cs.pairs[:max_preview]:
        print(f"  {hw!r} -> {name!r}")

print("\n=== Step 4: Parse channel-name arrays (hardware + human) ===")
fast_digital = ChannelSet("Fast digital", parse_channel_pairs_from_cluster_array(get_named_array(root, "Fast digital names")))
slow_digital = ChannelSet("Slow digital", parse_channel_pairs_from_cluster_array(get_named_array(root, "Slow digital names")))
fast_analogue = ChannelSet("Fast analogue", parse_channel_pairs_from_cluster_array(get_named_array(root, "Fast analogue names")))
slow_analogue = ChannelSet("Slow analogue", parse_channel_pairs_from_cluster_array(get_named_array(root, "Slow analogue names")))

summarize_channelset(fast_digital)
summarize_channelset(slow_digital)
summarize_channelset(fast_analogue)
summarize_channelset(slow_analogue)

# %% Parse Step headers (Sequence header top) + summary prints
def _parse_bool01(e: Optional[ET.Element]) -> Optional[int]:
    if e is None:
        return None
    v = txt(e.find("lv:Val", NS))
    if v == "":
        return None
    try:
        return int(v)
    except ValueError:
        return None

def _parse_int(e: Optional[ET.Element]) -> Optional[int]:
    if e is None:
        return None
    v = txt(e.find("lv:Val", NS))
    if v == "":
        return None
    try:
        return int(v)
    except ValueError:
        return None

def _parse_float(e: Optional[ET.Element]) -> Optional[float]:
    if e is None:
        return None
    v = txt(e.find("lv:Val", NS))
    if v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None

def _parse_time_unit(ew: Optional[ET.Element]) -> str:
    """
    LV enum (EW): has <Choice>... and a <Val> index.
    Your sample: Choices = [µs, ms, s], Val=1 -> 'ms' (so treat <Val> as 0-based index).
    """
    if ew is None:
        return ""
    choices = [txt(c) for c in ew.findall("lv:Choice", NS)]
    idx_txt = txt(ew.find("lv:Val", NS))
    try:
        idx = int(idx_txt)
    except ValueError:
        idx = None

    if idx is not None and 0 <= idx < len(choices):
        return choices[idx]
    return choices[0] if choices else idx_txt

def _dt_to_seconds(dt_value: Optional[float], dt_unit: str) -> Optional[float]:
    if dt_value is None:
        return None
    u = dt_unit.strip()
    if u in ("µs", "us", "µsec", "microsecond", "microseconds"):
        return dt_value * 1e-6
    if u in ("ms", "msec", "millisecond", "milliseconds"):
        return dt_value * 1e-3
    if u in ("s", "sec", "second", "seconds"):
        return dt_value
    if u in ("min", "minute", "minutes"):
        return dt_value * 60.0
    return None

def parse_step_headers(root: ET.Element) -> list[StepHeader]:
    header_arr = get_named_array(root, "Sequence header top")
    dims = array_dims(header_arr)
    clusters = header_arr.findall("lv:Cluster", NS)

    print("\n=== Step 5: Parse step headers (Sequence header top) ===")
    print("Header array dims:", dims)
    print("Header cluster count:", len(clusters))

    steps: list[StepHeader] = []
    for i, cl in enumerate(clusters):
        # Strings
        event_name = ""
        step_name = ""
        for s in cl.findall("lv:String", NS):
            key = txt(s.find("lv:Name", NS))
            val = txt(s.find("lv:Val", NS))
            if key == "Event name":
                event_name = val
            elif key == "Time step name":
                step_name = val

        # DBL time step length
        dt_len_el = next(
            (d for d in cl.findall("lv:DBL", NS) if txt(d.find("lv:Name", NS)) == "Time step length"),
            None,
        )
        dt_value = _parse_float(dt_len_el)

        # EW time unit
        ew = next((e for e in cl.findall("lv:EW", NS) if txt(e.find("lv:Name", NS)) == "Time unit"), None)
        dt_unit = _parse_time_unit(ew)
        dt_seconds = _dt_to_seconds(dt_value, dt_unit)

        # Event ID
        ev_id_el = next((x for x in cl.findall("lv:I32", NS) if txt(x.find("lv:Name", NS)) == "Event ID"), None)
        event_id = _parse_int(ev_id_el)

        # Booleans (0/1)
        hide_el = next((b for b in cl.findall("lv:Boolean", NS) if txt(b.find("lv:Name", NS)) == "Hide event steps"), None)
        pop_el  = next((b for b in cl.findall("lv:Boolean", NS) if txt(b.find("lv:Name", NS)) == "Populate multirun"), None)
        skip_el = next((b for b in cl.findall("lv:Boolean", NS) if txt(b.find("lv:Name", NS)) == "Skip Step"), None)

        steps.append(
            StepHeader(
                idx=i,
                event_name=event_name,
                step_name=step_name,
                dt_value=float(dt_value) if dt_value is not None else float("nan"),
                dt_unit=dt_unit,
                dt_seconds=dt_seconds,
                event_id=event_id,
                hide_event_steps=_parse_bool01(hide_el),
                populate_multirun=_parse_bool01(pop_el),
                skip_step=_parse_bool01(skip_el),
            )
        )

    # Summary prints
    print("\n--- Step header summary ---")
    print("Total steps:", len(steps))
    unique_events = sorted({s.event_name for s in steps if s.event_name})
    print("Unique events:", len(unique_events))
    if unique_events:
        print("First events:", unique_events[:6])

    bad_units = [s for s in steps if s.dt_seconds is None]
    if bad_units:
        print(f"WARNING: {len(bad_units)} steps have unknown/unsupported time units (dt_seconds=None).")

    total_time = sum((s.dt_seconds or 0.0) for s in steps)
    print("Total sequence time (computed):", total_time, "seconds")
    print("First 8 headers:")
    for s in steps[:8]:
        print(
            f"  [{s.idx:02d}] event={s.event_name!r} step={s.step_name!r} "
            f"dt={s.dt_value:g}{s.dt_unit} ({s.dt_seconds if s.dt_seconds is not None else '??'} s) "
            f"event_id={s.event_id} skip={s.skip_step}"
        )
    return steps

steps = parse_step_headers(root)

# %% Parse Digital channel matrices
def parse_boolean_array_2d(values_arr: ET.Element, channels_from_names: int, label: str) -> np.ndarray:
    dims = array_dims(values_arr)
    bool_nodes = values_arr.findall("lv:Boolean", NS)
    total_items = len(bool_nodes)

    print(f"\n=== Step 6: Parse {label} (digital bool matrix) ===")
    print("Dimsize in file:", dims)
    print("Channel count from names:", channels_from_names)
    print("Total boolean items:", total_items)

    if channels_from_names <= 0:
        raise ValueError(f"{label}: channels_from_names is {channels_from_names}, can't proceed.")
    if total_items % channels_from_names != 0:
        raise ValueError(f"{label}: total_items ({total_items}) not divisible by channels ({channels_from_names}).")

    inferred_steps = total_items // channels_from_names
    print("Inferred steps from counts:", inferred_steps)

    flat = np.fromiter(
        (int(txt(b.find("lv:Val", NS))) for b in bool_nodes),
        dtype=np.uint8,
        count=total_items,
    ).astype(bool)

    if len(dims) == 2 and dims[0] == channels_from_names:
        mat = flat.reshape((dims[0], dims[1]), order="C")
        print("Reshape used: (channels, steps) from dimsize:", (dims[0], dims[1]))
    elif len(dims) == 2 and dims[1] == channels_from_names:
        mat = flat.reshape((dims[0], dims[1]), order="C").T
        print("Reshape used: transpose of dimsize:", (dims[0], dims[1]), "->", mat.shape)
    else:
        mat = flat.reshape((channels_from_names, inferred_steps), order="C")
        print("Reshape used: fallback (channels, inferred_steps):", mat.shape)

    print("Final matrix shape:", mat.shape, "dtype:", mat.dtype)
    print("Preview [ch0, :16]:", mat[0, :16].astype(int).tolist())
    return mat

fast_dig_matrix = parse_boolean_array_2d(get_named_array(root, "Fast digital channels"), fast_digital.count, "Fast digital channels")
slow_dig_matrix = parse_boolean_array_2d(get_named_array(root, "Slow digital channels"), slow_digital.count, "Slow digital channels")
digital_mats = DigitalMatrices(fast=fast_dig_matrix, slow=slow_dig_matrix)

# Sanity check: digital steps should match header steps (often 55)
if steps:
    if digital_mats.fast.shape[1] != len(steps):
        print(f"WARNING: fast digital steps={digital_mats.fast.shape[1]} but header steps={len(steps)}")
    if digital_mats.slow.shape[1] != len(steps):
        print(f"WARNING: slow digital steps={digital_mats.slow.shape[1]} but header steps={len(steps)}")

# %% Parse Analogue channel matrices
def parse_analogue_cluster_voltage_and_ramp(cluster: ET.Element) -> tuple[float, bool]:
    """
    Per your format:
      <Boolean><Name>Ramp?</Name><Val>0/1</Val></Boolean>
      <DBL><Name>Voltage</Name><Val>2.067...</Val></DBL>
    """
    ramp_val: Optional[bool] = None
    volt_val: Optional[float] = None

    for b in cluster.findall("lv:Boolean", NS):
        if txt(b.find("lv:Name", NS)) == "Ramp?":
            ramp_val = bool(int(txt(b.find("lv:Val", NS)) or "0"))
            break

    for d in cluster.findall("lv:DBL", NS):
        if txt(d.find("lv:Name", NS)) == "Voltage":
            vtxt = txt(d.find("lv:Val", NS))
            volt_val = float(vtxt) if vtxt else float("nan")
            break

    if ramp_val is None:
        ramp_val = False
    if volt_val is None:
        volt_val = float("nan")

    return volt_val, ramp_val

def parse_analogue_array_2d(values_arr: ET.Element, channels_from_names: int, label: str) -> tuple[np.ndarray, np.ndarray]:
    dims = array_dims(values_arr)
    clusters = values_arr.findall("lv:Cluster", NS)
    total_items = len(clusters)

    print(f"\n=== Step 7: Parse {label} (analogue → voltage float + ramp bool) ===")
    print("Dimsize in file:", dims)
    print("Channel count from names:", channels_from_names)
    print("Total cluster items:", total_items)

    if channels_from_names <= 0:
        raise ValueError(f"{label}: channels_from_names is {channels_from_names}, can't proceed.")
    if total_items % channels_from_names != 0:
        raise ValueError(f"{label}: total_items ({total_items}) not divisible by channels ({channels_from_names}).")

    inferred_steps = total_items // channels_from_names
    print("Inferred steps from counts:", inferred_steps)

    volt_flat = np.empty(total_items, dtype=float)
    ramp_flat = np.empty(total_items, dtype=bool)

    for i, cl in enumerate(clusters):
        v, r = parse_analogue_cluster_voltage_and_ramp(cl)
        volt_flat[i] = v
        ramp_flat[i] = r

    if clusters:
        v0, r0 = parse_analogue_cluster_voltage_and_ramp(clusters[0])
        print("Example first cluster parsed -> Voltage:", v0, "Ramp?:", int(r0))

    def reshape_like(flat: np.ndarray) -> np.ndarray:
        if len(dims) == 2 and dims[0] == channels_from_names:
            return flat.reshape((dims[0], dims[1]), order="C")
        if len(dims) == 2 and dims[1] == channels_from_names:
            return flat.reshape((dims[0], dims[1]), order="C").T
        return flat.reshape((channels_from_names, inferred_steps), order="C")

    volt = reshape_like(volt_flat)
    ramp = reshape_like(ramp_flat)

    print("Final voltage shape:", volt.shape, "dtype:", volt.dtype)
    print("Final ramp shape   :", ramp.shape, "dtype:", ramp.dtype)
    print("Voltage preview [ch0, :8]:", volt[0, :8].tolist())
    print("Ramp preview    [ch0, :16]:", ramp[0, :16].astype(int).tolist())
    return volt, ramp

fast_volt, fast_ramp = parse_analogue_array_2d(get_named_array(root, "Fast analogue array"), fast_analogue.count, "Fast analogue array")
slow_volt, slow_ramp = parse_analogue_array_2d(get_named_array(root, "Slow analogue array"), slow_analogue.count, "Slow analogue array")

analogue_mats = AnalogueMatrices(
    fast_voltage=fast_volt,
    fast_ramp=fast_ramp,
    slow_voltage=slow_volt,
    slow_ramp=slow_ramp,
)

# Sanity check: analogue steps should match header steps too
if steps:
    if analogue_mats.fast_voltage.shape[1] != len(steps):
        print(f"WARNING: fast analogue steps={analogue_mats.fast_voltage.shape[1]} but header steps={len(steps)}")
    if analogue_mats.slow_voltage.shape[1] != len(steps):
        print(f"WARNING: slow analogue steps={analogue_mats.slow_voltage.shape[1]} but header steps={len(steps)}")

# %% Quick final sanity summary (including access patterns)
print("\n=== Step 8: Final sanity summary ===")
print("Header steps:", len(steps))
print(f"Fast digital:  {digital_mats.fast.shape} (channels x steps). Example: {fast_digital.count} x {digital_mats.fast.shape[1]}")
print(f"Slow digital:  {digital_mats.slow.shape} (channels x steps). Example: {slow_digital.count} x {digital_mats.slow.shape[1]}")
print(f"Fast analogue: {analogue_mats.fast_voltage.shape} (channels x steps) + ramp {analogue_mats.fast_ramp.shape}")
print(f"Slow analogue: {analogue_mats.slow_voltage.shape} (channels x steps) + ramp {analogue_mats.slow_ramp.shape}")

ch = 0
step = 0
print("\nExample indexing (channel_index, step_index):")
print("Header[0]:", steps[0] if steps else None)
print("Fast digital value:", int(digital_mats.fast[ch, step]), "| channel:", fast_digital.pairs[ch])
print("Fast analogue voltage:", analogue_mats.fast_voltage[ch, step], "| ramp?:", int(analogue_mats.fast_ramp[ch, step]), "| channel:", fast_analogue.pairs[ch])

# %% Plot: Fast digital matrix (log-width cells, colored 0/1)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

print("=== Plot: Fast digital channels matrix ===")

# --- pull data + labels, and align lengths safely ---
mat = digital_mats.fast  # shape (channels, steps)
n_ch, n_steps_mat = mat.shape
n_steps_hdr = len(steps)

n_steps = min(n_steps_mat, n_steps_hdr) if n_steps_hdr else n_steps_mat
if n_steps_hdr and n_steps_mat != n_steps_hdr:
    print(f"WARNING: fast digital has {n_steps_mat} steps, headers have {n_steps_hdr}. Plotting first {n_steps}.")

mat_i = mat[:, :n_steps].astype(int)
y_labels = fast_digital.human_names[:n_ch] if len(fast_digital.human_names) >= n_ch else [str(i) for i in range(n_ch)]
x_labels = [steps[i].step_name for i in range(n_steps)] if n_steps_hdr else [str(i) for i in range(n_steps)]

# --- compute log-widths from step durations (seconds) ---
# user example: make 1µs be ~1/6 the width of 1s
# linear map in log10 space: w = (5/6)*log10(t) + 6
# so w(1s)=6 and w(1e-6s)=1
dt_s = np.ones(n_steps, dtype=float)
if n_steps_hdr:
    for i in range(n_steps):
        v = steps[i].dt_seconds
        dt_s[i] = float(v) if (v is not None and v > 0) else 1e-6  # fallback to 1µs if unknown/invalid

dt_s = np.clip(dt_s, 1e-12, None)
w = (5.0 / 6.0) * np.log10(dt_s) + 6.0
# w = np.clip(w, 0.25, None)  # keep every step visible

x_edges = np.concatenate([[0.0], np.cumsum(w)])
y_edges = np.arange(n_ch + 1, dtype=float)

x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
y_centers = np.arange(n_ch, dtype=float) + 0.5

print(f"Channels: {n_ch}, Steps: {n_steps}")
print("Duration range (s):", float(np.min(dt_s)), "→", float(np.max(dt_s)))
print("Width range (arb.):", float(np.min(w)), "→", float(np.max(w)))

# --- plot ---
cmap = ListedColormap(["darkred", "lime"])  # 0 -> dark red, 1 -> bright green
norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

fig, ax = plt.subplots(figsize=(18, 10), constrained_layout=True)

# pcolormesh supports non-uniform cell widths via custom x_edges
pm = ax.pcolormesh(
    x_edges,
    y_edges,
    mat_i,
    cmap=cmap,
    norm=norm,
    shading="flat",
    linewidth=0.05,
    edgecolors="black",
)

ax.set_title("Fast digital channels (0=dark red, 1=bright green) with log-width time steps")
ax.set_xlabel("Step (cell width ~ log10(dt_seconds))")
ax.set_ylabel("Channel index / name")

# ticks + tiny labels
ax.set_xticks(x_centers)
ax.set_xticklabels(x_labels, rotation=90, fontsize=7)
ax.set_yticks(y_centers)
ax.set_yticklabels([f"{i:02d} {name}" for i, name in enumerate(y_labels)], fontsize=7)

# optional: put channel 0 at top (often nicer for "matrix" views)
ax.invert_yaxis()

plt.show()
# %%


# ============================================================
# Width computation (independent from plotting)
# ============================================================

def _safe_dt_seconds(steps, n_steps: int) -> np.ndarray:
    """Extract dt_seconds for the first n_steps; assumes positive/non-zero but guards anyway."""
    dt = np.empty(n_steps, dtype=float)
    for i in range(n_steps):
        v = steps[i].dt_seconds
        dt[i] = float(v) if (v is not None and v > 0) else 1e-6
    return dt

def _affine_normalize(x: np.ndarray, out_min: float, out_max: float) -> np.ndarray:
    """Map x -> [out_min, out_max] (handles constant arrays)."""
    x = np.asarray(x, dtype=float)
    xmin, xmax = float(np.min(x)), float(np.max(x))
    if np.isclose(xmax, xmin):
        return np.full_like(x, (out_min + out_max) * 0.5)
    t = (x - xmin) / (xmax - xmin)
    return out_min + t * (out_max - out_min)

def compute_step_widths_constant(n_steps: int, width: float = 1.0) -> np.ndarray:
    return np.full(n_steps, float(width), dtype=float)

def compute_step_widths_log_one_stage(
    dt_seconds: np.ndarray,
    max_ratio: float = 8.0,
    min_width: float = 1.0,
    log_base: float = 10.0,
) -> np.ndarray:
    """
    One-stage log: widths are monotonic with log(dt), but *bounded* so max/min <= max_ratio.
    """
    dt = np.asarray(dt_seconds, dtype=float)
    dt = np.clip(dt, 1e-300, None)
    raw = np.log(dt) / np.log(log_base)
    max_width = min_width * float(max_ratio)
    return _affine_normalize(raw, min_width, max_width)

def compute_event_spans(steps, n_steps: int):
    """
    Returns contiguous event spans: [(start_idx, end_idx_exclusive, event_name), ...]
    """
    spans = []
    if n_steps <= 0:
        return spans

    start = 0
    cur = steps[0].event_name
    for i in range(1, n_steps):
        ev = steps[i].event_name
        if ev != cur:
            spans.append((start, i, cur))
            start, cur = i, ev
    spans.append((start, n_steps, cur))
    return spans

def compute_step_widths_log_two_stage(
    steps,
    n_steps: int,
    max_total_ratio: float = 20.0,
    min_width: float = 1.0,
    log_base: float = 10.0,
) -> np.ndarray:
    """
    Two-stage log with a controlled overall width ratio.

    - Event widths: log(duration) mapped to [min_width, min_width*event_ratio]
    - Step widths within each event: log(dt) mapped to [1, step_ratio], then scaled
      so the steps in the event sum to that event's width.

    With event_ratio * step_ratio ~= max_total_ratio (by default split as sqrt).
    """
    if n_steps <= 0:
        return np.array([], dtype=float)

    dt = _safe_dt_seconds(steps, n_steps)
    spans = compute_event_spans(steps, n_steps)

    # split the allowed overall ratio across the two stages
    event_ratio = float(np.sqrt(max_total_ratio))
    step_ratio = float(max_total_ratio) / event_ratio

    # --- event widths ---
    event_durs = np.array([np.sum(dt[s:e]) for (s, e, _) in spans], dtype=float)
    event_raw = np.log(event_durs) / np.log(log_base)
    event_widths = _affine_normalize(event_raw, min_width, min_width * event_ratio)

    # --- step widths within each event (scaled to event width) ---
    widths = np.empty(n_steps, dtype=float)
    for (event_idx, (s, e, _)) in enumerate(spans):
        dts = dt[s:e]
        step_raw = np.log(dts) / np.log(log_base)
        step_unscaled = _affine_normalize(step_raw, 1.0, step_ratio)  # bounded within-event
        # scale to match the event width exactly
        scale = event_widths[event_idx] / float(np.sum(step_unscaled))
        widths[s:e] = step_unscaled * scale

    return widths

# ============================================================
# Plotting (takes widths as input)
# ============================================================

def plot_fast_digital_grid(
    mat_bool: np.ndarray,            # shape (channels, steps)
    channel_names: list[str],        # y labels, length channels
    steps,                           # list[StepHeader] (for labels + events)
    step_widths: np.ndarray,         # length n_steps_to_plot
    label_fontsize: int = 7,
):
    print("=== Plot: Fast digital channels matrix ===")

    n_ch, n_steps_mat = mat_bool.shape
    n_steps_hdr = len(steps) if steps is not None else 0

    n_steps = min(n_steps_mat, n_steps_hdr, len(step_widths)) if n_steps_hdr else min(n_steps_mat, len(step_widths))
    if n_steps <= 0:
        raise ValueError("No steps available to plot (check matrix / headers / widths).")

    if n_steps_hdr and n_steps_mat != n_steps_hdr:
        print(f"WARNING: matrix has {n_steps_mat} steps, headers have {n_steps_hdr}. Plotting first {n_steps}.")
    if len(step_widths) != n_steps:
        step_widths = np.asarray(step_widths, dtype=float)[:n_steps]

    # data as 0/1 ints for colormap boundary norm
    mat_i = mat_bool[:, :n_steps].astype(int)

    # labels
    y_labels = channel_names[:n_ch] if len(channel_names) >= n_ch else [str(i) for i in range(n_ch)]
    x_step_labels = [steps[i].step_name for i in range(n_steps)] if n_steps_hdr else [str(i) for i in range(n_steps)]

    # edges/centers from precomputed widths
    w = np.asarray(step_widths, dtype=float)
    if np.any(w <= 0):
        raise ValueError("All step widths must be positive.")
    x_edges = np.concatenate([[0.0], np.cumsum(w)])
    y_edges = np.arange(n_ch + 1, dtype=float)

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = np.arange(n_ch, dtype=float) + 0.5

    # colors
    cmap = ListedColormap(["darkred", "lime"])  # 0 -> dark red, 1 -> bright green
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

    fig, ax = plt.subplots(figsize=(18, 10), constrained_layout=True)
    ax.pcolormesh(
        x_edges, y_edges, mat_i,
        cmap=cmap, norm=norm,
        shading="flat",
        linewidth=0.05,
        edgecolors="black",
    )

    # ax.set_title(title)
    ax.set_xlabel("Steps (cell widths provided externally)")
    ax.set_ylabel("Channel index / name")

    ax.set_xticks(x_centers)
    ax.set_xticklabels(x_step_labels, rotation=90, fontsize=label_fontsize)
    ax.set_yticks(y_centers)
    ax.set_yticklabels([f"{i:02d} {name}" for i, name in enumerate(y_labels)], fontsize=label_fontsize)
    ax.invert_yaxis()

    # --- event braces + vertical event labels on top ---
    if n_steps_hdr:
        spans = compute_event_spans(steps, n_steps)
        ax_top = ax.twiny()
        ax_top.set_xlim(ax.get_xlim())
        ax_top.set_xticks([])  # we draw our own labels/braces
        # ax_top.set_xlabel("Events")

        y0 = 1.00
        h = 0.03
        for (s, e, ev_name) in spans:
            x0, x1 = float(x_edges[s]), float(x_edges[e])
            # brace
            ax_top.plot(
                [x0, x0, x1, x1],
                [y0, y0 + h, y0 + h, y0],
                transform=ax_top.get_xaxis_transform(),
                color="black",
                lw=0.8,
                clip_on=False,
            )
            # vertical label
            ax_top.text(
                0.5 * (x0 + x1), y0 + h + 0.01, ev_name,
                transform=ax_top.get_xaxis_transform(),
                ha="left", va="bottom",
                fontsize=label_fontsize,
                rotation=90, rotation_mode="anchor",
                clip_on=False,
            )

    plt.show()
    return fig, ax

# ============================================================
# Example usage (choose a width mode)
# ============================================================

# Align lengths (matrix vs headers) for width computation
n_steps_for_widths = min(digital_mats.fast.shape[1], len(steps))

# Mode 1: constant widths
# widths = compute_step_widths_constant(n_steps_for_widths, width=1.0)

# Mode 2: one-stage log (bounded by max_ratio)
# dt = _safe_dt_seconds(steps, n_steps_for_widths)
# widths = compute_step_widths_log_one_stage(dt, max_ratio=8.0, min_width=1.0)

# Mode 3: two-stage log (bounded overall by max_total_ratio)
widths = compute_step_widths_log_two_stage(steps, n_steps_for_widths, max_total_ratio=20.0, min_width=0.1)

# Plot using externally computed widths
plot_fast_digital_grid(
    mat_bool=digital_mats.fast,
    channel_names=fast_digital.human_names,
    steps=steps,
    step_widths=widths,
    label_fontsize=7,
)


# %%
#########
""" Analogue ramp format. This gives us the voltage and ramp boolean at the same time
          <{http://www.ni.com/LVData}Name> attrs={} text='Analogue cluster'
          <{http://www.ni.com/LVData}NumElts> attrs={} text='2'
          <{http://www.ni.com/LVData}Boolean> attrs={} text=''
            <{http://www.ni.com/LVData}Name> attrs={} text='Ramp?'
            <{http://www.ni.com/LVData}Val> attrs={} text='0'
          <{http://www.ni.com/LVData}DBL> attrs={} text=''
            <{http://www.ni.com/LVData}Name> attrs={} text='Voltage'
            <{http://www.ni.com/LVData}Val> attrs={} text='2.06700000000000'
        <{http://www.ni.com/LVData}Cluster> attrs={} text=''

The value stated after 'Ramp?' is the boolean related to whether we ramp into the next step.
The value stated after 'Voltage' is the voltage float we care about.
"""


"""  Analogue name format (the same for slow analogues):
<{http://www.ni.com/LVData}Name> attrs={} text='Fast analogue names'
        <{http://www.ni.com/LVData}Dimsize> attrs={} text='8'
        <{http://www.ni.com/LVData}Cluster> attrs={} text=''
          <{http://www.ni.com/LVData}Name> attrs={} text='Channel names'
          <{http://www.ni.com/LVData}NumElts> attrs={} text='2'
          <{http://www.ni.com/LVData}String> attrs={} text=''
            <{http://www.ni.com/LVData}Name> attrs={} text='Hardware ID'
            <{http://www.ni.com/LVData}Val> attrs={} text='FAO 0'
          <{http://www.ni.com/LVData}String> attrs={} text=''
            <{http://www.ni.com/LVData}Name> attrs={} text='Name'
            <{http://www.ni.com/LVData}Val> attrs={} text='Cs Cool AOM Freq'
        <{http://www.ni.com/LVData}Cluster> attrs={} text=''
          <{http://www.ni.com/LVData}Name> attrs={} text='Channel names'
          <{http://www.ni.com/LVData}NumElts> attrs={} text='2'
          <{http://www.ni.com/LVData}String> attrs={} text=''
            <{http://www.ni.com/LVData}Name> attrs={} text='Hardware ID'
            <{http://www.ni.com/LVData}Val> attrs={} text='FAO 1'
          <{http://www.ni.com/LVData}String> attrs={} text=''
            <{http://www.ni.com/LVData}Name> attrs={} text='Name'
            <{http://www.ni.com/LVData}Val> attrs={} text='817 shallow-angle power'
        <{http://www.ni.com/LVData}Cluster> attrs={} text=''
"""


""" Fast digital value formats (one big matrix) (same for slow channels)
<{http://www.ni.com/LVData}Name> attrs={} text='Fast digital channels'
        <{http://www.ni.com/LVData}Dimsize> attrs={} text='56'
        <{http://www.ni.com/LVData}Dimsize> attrs={} text='55'
        <{http://www.ni.com/LVData}Boolean> attrs={} text=''
          <{http://www.ni.com/LVData}Name> attrs={} text='Boolean'
          <{http://www.ni.com/LVData}Val> attrs={} text='0'
        <{http://www.ni.com/LVData}Boolean> attrs={} text=''
          <{http://www.ni.com/LVData}Name> attrs={} text='Boolean'
          <{http://www.ni.com/LVData}Val> attrs={} text='0'
        <{http://www.ni.com/LVData}Boolean> attrs={} text=''
          <{http://www.ni.com/LVData}Name> attrs={} text='Boolean'
          <{http://www.ni.com/LVData}Val> attrs={} text='0'
        <{http://www.ni.com/LVData}Boolean> attrs={} text=''
          <{http://www.ni.com/LVData}Name> attrs={} text='Boolean'
          <{http://www.ni.com/LVData}Val> attrs={} text='0'
        <{http://www.ni.com/LVData}Boolean> attrs={} text=''
          <{http://www.ni.com/LVData}Name> attrs={} text='Boolean'
          <{http://www.ni.com/LVData}Val> attrs={} text='0'
        <{http://www.ni.com/LVData}Boolean> attrs={} text=''
"""

""" Fast (slow) digital names format
<{http://www.ni.com/LVData}Name> attrs={} text='Fast digital names'
        <{http://www.ni.com/LVData}Dimsize> attrs={} text='56'
        <{http://www.ni.com/LVData}Cluster> attrs={} text=''
          <{http://www.ni.com/LVData}Name> attrs={} text='Channel names'
          <{http://www.ni.com/LVData}NumElts> attrs={} text='2'
          <{http://www.ni.com/LVData}String> attrs={} text=''
            <{http://www.ni.com/LVData}Name> attrs={} text='Hardware ID'
            <{http://www.ni.com/LVData}Val> attrs={} text='FDO 0'
          <{http://www.ni.com/LVData}String> attrs={} text=''
            <{http://www.ni.com/LVData}Name> attrs={} text='Name'
            <{http://www.ni.com/LVData}Val> attrs={} text='977 P0'
        <{http://www.ni.com/LVData}Cluster> attrs={} text=''
          <{http://www.ni.com/LVData}Name> attrs={} text='Channel names'
          <{http://www.ni.com/LVData}NumElts> attrs={} text='2'
          <{http://www.ni.com/LVData}String> attrs={} text=''
            <{http://www.ni.com/LVData}Name> attrs={} text='Hardware ID'
            <{http://www.ni.com/LVData}Val> attrs={} text='FDO 1'
          <{http://www.ni.com/LVData}String> attrs={} text=''
            <{http://www.ni.com/LVData}Name> attrs={} text='Name'
            <{http://www.ni.com/LVData}Val> attrs={} text='977 P1'
        <{http://www.ni.com/LVData}Cluster> attrs={} text=''

Here, we define are pairs [('FDO 0', '977 P0'), ('FDO1', '977 P1'), ...]
Good to store the 'hardware names' but we can just access this with l[1] -> '977 P1' for example
"""

"""
step names
<{http://www.ni.com/LVData}Cluster> attrs={} text=''
      <{http://www.ni.com/LVData}Name> attrs={} text='Experimental sequence cluster'
      <{http://www.ni.com/LVData}NumElts> attrs={} text='10'
      <{http://www.ni.com/LVData}Array> attrs={} text=''
        <{http://www.ni.com/LVData}Name> attrs={} text='Sequence header top'
        <{http://www.ni.com/LVData}Dimsize> attrs={} text='55'
        <{http://www.ni.com/LVData}Cluster> attrs={} text=''
          <{http://www.ni.com/LVData}Name> attrs={} text='Sequence header cluster'
          <{http://www.ni.com/LVData}NumElts> attrs={} text='10'
          <{http://www.ni.com/LVData}String> attrs={} text=''
            <{http://www.ni.com/LVData}Name> attrs={} text='Event name'
            <{http://www.ni.com/LVData}Val> attrs={} text='RbMOTtoTweezers'
          <{http://www.ni.com/LVData}DBL> attrs={} text=''
            <{http://www.ni.com/LVData}Name> attrs={} text='Time step length'
            <{http://www.ni.com/LVData}Val> attrs={} text='1.00000000000000'
          <{http://www.ni.com/LVData}Cluster> attrs={} text=''
            <{http://www.ni.com/LVData}Name> attrs={} text='Trigger details'
            <{http://www.ni.com/LVData}NumElts> attrs={} text='4'
            <{http://www.ni.com/LVData}DBL> attrs={} text=''
            <{http://www.ni.com/LVData}EW> attrs={} text=''
            <{http://www.ni.com/LVData}U8> attrs={} text=''
            <{http://www.ni.com/LVData}Boolean> attrs={} text=''
          <{http://www.ni.com/LVData}Cluster> attrs={} text=''
            <{http://www.ni.com/LVData}Name> attrs={} text='GPIB routine data'
            <{http://www.ni.com/LVData}NumElts> attrs={} text='2'
            <{http://www.ni.com/LVData}EW> attrs={} text=''
            <{http://www.ni.com/LVData}Boolean> attrs={} text=''
          <{http://www.ni.com/LVData}String> attrs={} text=''
            <{http://www.ni.com/LVData}Name> attrs={} text='Time step name'
            <{http://www.ni.com/LVData}Val> attrs={} text='RbIntiate'
          <{http://www.ni.com/LVData}Boolean> attrs={} text=''
            <{http://www.ni.com/LVData}Name> attrs={} text='Hide event steps'
            <{http://www.ni.com/LVData}Val> attrs={} text='1'
          <{http://www.ni.com/LVData}Boolean> attrs={} text=''
            <{http://www.ni.com/LVData}Name> attrs={} text='Populate multirun'
            <{http://www.ni.com/LVData}Val> attrs={} text='1'
          <{http://www.ni.com/LVData}EW> attrs={} text=''
            <{http://www.ni.com/LVData}Name> attrs={} text='Time unit'
            <{http://www.ni.com/LVData}Choice> attrs={} text='µs'
            <{http://www.ni.com/LVData}Choice> attrs={} text='ms'
            <{http://www.ni.com/LVData}Choice> attrs={} text='s'
            <{http://www.ni.com/LVData}Val> attrs={} text='1'
          <{http://www.ni.com/LVData}I32> attrs={} text=''
            <{http://www.ni.com/LVData}Name> attrs={} text='Event ID'
            <{http://www.ni.com/LVData}Val> attrs={} text='0'
          <{http://www.ni.com/LVData}Boolean> attrs={} text=''
            <{http://www.ni.com/LVData}Name> attrs={} text='Skip Step'
            <{http://www.ni.com/LVData}Val> attrs={} text='0'
        <{http://www.ni.com/LVData}Cluster> attrs={} text=''
          <{http://www.ni.com/LVData}Name> attrs={} text='Sequence header cluster'
          <{http://www.ni.com/LVData}NumElts> attrs={} text='10'
          <{http://www.ni.com/LVData}String> attrs={} text=''
            <{http://www.ni.com/LVData}Name> attrs={} text='Event name'
            <{http://www.ni.com/LVData}Val> attrs={} text='RbMOTtoTweezers'
          <{http://www.ni.com/LVData}DBL> attrs={} text=''
            <{http://www.ni.com/LVData}Name> attrs={} text='Time step length'
            <{http://www.ni.com/LVData}Val> attrs={} text='150.00000000000000'
          <{http://www.ni.com/LVData}Cluster> attrs={} text=''
            <{http://www.ni.com/LVData}Name> attrs={} text='Trigger details'
            <{http://www.ni.com/LVData}NumElts> attrs={} text='4'
            <{http://www.ni.com/LVData}DBL> attrs={} text=''
            <{http://www.ni.com/LVData}EW> attrs={} text=''
            <{http://www.ni.com/LVData}U8> attrs={} text=''
            <{http://www.ni.com/LVData}Boolean> attrs={} text=''
          <{http://www.ni.com/LVData}Cluster> attrs={} text=''
            <{http://www.ni.com/LVData}Name> attrs={} text='GPIB routine data'
            <{http://www.ni.com/LVData}NumElts> attrs={} text='2'
            <{http://www.ni.com/LVData}EW> attrs={} text=''
            <{http://www.ni.com/LVData}Boolean> attrs={} text=''
          <{http://www.ni.com/LVData}String> attrs={} text=''
            <{http://www.ni.com/LVData}Name> attrs={} text='Time step name'
            <{http://www.ni.com/LVData}Val> attrs={} text='RbMOT'
          <{http://www.ni.com/LVData}Boolean> attrs={} text=''
            <{http://www.ni.com/LVData}Name> attrs={} text='Hide event steps'
            <{http://www.ni.com/LVData}Val> attrs={} text='1'
          <{http://www.ni.com/LVData}Boolean> attrs={} text=''
            <{http://www.ni.com/LVData}Name> attrs={} text='Populate multirun'
            <{http://www.ni.com/LVData}Val> attrs={} text='1'
          <{http://www.ni.com/LVData}EW> attrs={} text=''
            <{http://www.ni.com/LVData}Name> attrs={} text='Time unit'
            <{http://www.ni.com/LVData}Choice> attrs={} text='µs'
            <{http://www.ni.com/LVData}Choice> attrs={} text='ms'
            <{http://www.ni.com/LVData}Choice> attrs={} text='s'
            <{http://www.ni.com/LVData}Val> attrs={} text='1'
          <{http://www.ni.com/LVData}I32> attrs={} text=''
            <{http://www.ni.com/LVData}Name> attrs={} text='Event ID'
            <{http://www.ni.com/LVData}Val> attrs={} text='0'
          <{http://www.ni.com/LVData}Boolean> attrs={} text=''
            <{http://www.ni.com/LVData}Name> attrs={} text='Skip Step'
            <{http://www.ni.com/LVData}Val> attrs={} text='0'
        <{http://www.ni.com/LVData}Cluster> attrs={} text=''
          <{http://www.ni.com/LVData}Name> attrs={} text='Sequence header cluster'
          <{http://www.ni.com/LVData}NumElts> attrs={} text='10'
          <{http://www.ni.com/LVData}String> attrs={} text=''
            <{http://www.ni.com/LVData}Name> attrs={} text='Event name'
            <{http://www.ni.com/LVData}Val> attrs={} text='RbMOTtoTweezers'
          <{http://www.ni.com/LVData}DBL> attrs={} text=''
            <{http://www.ni.com/LVData}Name> attrs={} text='Time step length'
            <{http://www.ni.com/LVData}Val> attrs={} text='1.00000000000000'
          <{http://www.ni.com/LVData}Cluster> attrs={} text=''
            <{http://www.ni.com/LVData}Name> attrs={} text='Trigger details'
            <{http://www.ni.com/LVData}NumElts> attrs={} text='4'
            <{http://www.ni.com/LVData}DBL> attrs={} text=''
"""
# %%
