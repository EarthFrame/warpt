# Tenstorrent Backend Integration

## Overview

This branch (`backend/tenstorrent`) adds Tenstorrent accelerator support to warpt. The implementation uses **Linux sysfs only** — no third-party SDK or pip dependencies are required. The Tenstorrent kernel-mode driver (tt-kmd) must be installed and loaded.

## Files Created / Modified

| File | Description |
|------|-------------|
| `warpt/backends/tenstorrent.py` | Accelerator backend — device discovery, telemetry, and device info via sysfs |
| `warpt/backends/power/tenstorrent_power.py` | Power monitoring backend — power draw, power limits, temperature, voltage, current |
| `warpt/backends/factory.py` | Modified to add Tenstorrent to the auto-detection chain (priority: NVIDIA > Tenstorrent > AMD > Intel) |
| `tests/test_tenstorrent_backend.py` | 45 mock-based tests (no hardware required to run) |

## What It Reads

### From `/sys/class/tenstorrent/tenstorrent!N/`

| Sysfs Attribute | Used For |
|-----------------|----------|
| `tt_card_type` | Product name (e.g. "n150", "n300") |
| `tt_serial` | Board serial number, used as device UUID |
| `tt_asic_id` | Chip-level ID (fallback UUID) |
| `tt_aiclk` | AI clock frequency (MHz) |
| `tt_arcclk` | ARC clock frequency (MHz) |
| `tt_axiclk` | AXI clock frequency (MHz) |
| `tt_fw_bundle_ver` | Firmware bundle version (reported as driver version) |
| `tt_arc_fw_ver` | ARC firmware version |
| `tt_eth_fw_ver` | Ethernet firmware version |
| `tt_m3app_fw_ver` | M3 app firmware version |

### From `/sys/class/hwmon/hwmonX/` (matched via PCI device symlink)

| Sysfs Attribute | Used For | Unit Conversion |
|-----------------|----------|-----------------|
| `name` | Chip architecture ("wormhole", "blackhole") | — |
| `temp1_input` | Temperature | millidegrees C → °C |
| `power1_input` | Power draw | microwatts → Watts |
| `power1_max` | Power limit (TDP) | microwatts → Watts |
| `in0_input` | Voltage | millivolts → Volts |
| `curr1_input` | Current | milliamps → Amps |

## What Is Not Available via sysfs

These fields return `None` or `0` as documented in the code:

- **Device memory** (`memory_gb = 0`) — sysfs does not expose DRAM capacity
- **Utilization** — no compute/memory utilization metrics
- **Throttle reasons** — not exposed
- **PCIe generation** — not directly exposed
- **N300 second ASIC** — the remote chip is connected via on-board ethernet and is not visible to sysfs

## How to Test

### 1. Run the unit tests (no hardware needed)

```bash
pip install -e ".[dev]"
pytest tests/test_tenstorrent_backend.py -v
```

All 45 tests are mock-based and should pass on any machine.

### 2. Test on a machine with Tenstorrent hardware

Verify tt-kmd is loaded:

```bash
ls /sys/class/tenstorrent/
# Should show: tenstorrent!0  tenstorrent!1  ...
```

Run warpt commands:

```bash
# List detected devices
warpt list

# Check power readings
warpt power
```

### 3. Things to verify on hardware

- [ ] Device discovery finds all expected cards
- [ ] `warpt list` shows correct card type, chip name, serial number, and firmware versions
- [ ] Temperature readings are reasonable (expected range: ~30-80°C idle/load)
- [ ] Power readings are reasonable (check against what `tt-smi` reports, if available)
- [ ] Clock frequencies match expected values
- [ ] On N300 cards: confirms 1 device is reported per card (the PCI-connected ASIC)
- [ ] On multi-card systems: all cards are discovered and indexed correctly

### 4. Quick sanity check via Python

```python
from warpt.backends.tenstorrent import TenstorrentBackend

backend = TenstorrentBackend()
print(f"Available: {backend.is_available()}")
print(f"Device count: {backend.get_device_count()}")

for dev in backend.list_devices():
    print(f"  [{dev.index}] {dev.model}")
    print(f"       UUID: {dev.uuid}")
    print(f"       Driver: {dev.driver_version}")
    print(f"       Extra: {dev.extra_metrics}")

for i in range(backend.get_device_count()):
    print(f"  [{i}] Temp: {backend.get_temperature(i)}°C")
    print(f"       Power: {backend.get_power_usage(i)}W")
```

## Not Yet Implemented

The following sysfs attributes from the SDK guide are not yet covered by the backend:

- **`tt_ttflash_ver`** — The tt-flash utility version. Noted as "may not be present on all cards" in the guide.
- **PCIe performance counters** (`pcie_perf_counters/` subdirectory) — Cumulative 32-bit counters for read/write data words across NOC 0 and NOC 1. The guide notes these "may not be present on all devices or firmware versions," and PCI counter accuracy is uncertain when DMA is used.

These can be added in a future pass if needed.

## Stress Testing

warpt includes 25 stress tests across 5 categories. Here's what can run on Tenstorrent systems:

### Non-GPU Tests (ready to run, no dependencies)

These tests don't use the GPU backend and will work on any Tenstorrent system:

```bash
warpt stress --category cpu       # Matrix multiply, zlib compression, hashing
warpt stress --category ram       # Bandwidth, latency, swap pressure
warpt stress --category storage   # Sequential/random read/write, mixed I/O
warpt stress --category network   # Loopback, point-to-point, bidirectional
```

### GPU Stress Tests (requires PyTorch + tt-metalium)

The 5 GPU stress tests (MatMul, FP64 Compute, Memory Bandwidth, Precision, CFD Simulation) use **PyTorch** to run compute workloads on the accelerator. The backend provides the device string (`"tt:0"`), and PyTorch handles execution.

**Requirement:** These tests will only work if **tt-metalium or tt-buda** is installed with PyTorch integration, so that `torch.device("tt:0")` is valid. Without it, the GPU tests will be skipped.

```bash
# If tt-metalium PyTorch integration is installed:
warpt stress --category accelerator
```

### Monitoring During Stress Tests

Even if the GPU stress tests run, some sidecar telemetry will be limited:

| Metric | Available? | Notes |
|--------|-----------|-------|
| TFLOPS / throughput | Yes | Measured by PyTorch directly |
| Temperature | Yes | From sysfs hwmon |
| Power draw | Yes | From sysfs hwmon |
| Memory usage | No | Not exposed by sysfs |
| Utilization % | No | Not exposed by sysfs |
| Throttle detection | No | Not exposed by sysfs |

Core benchmark numbers (TFLOPS, bandwidth) are unaffected — it's only the monitoring telemetry that's partial.

### Question for Testing

Do your test machines have tt-metalium or tt-buda installed with PyTorch integration? If yes, we'd appreciate results from running `warpt stress --category accelerator` as well.

## Feedback Requested

We would appreciate your input on the following:

1. **Sysfs attribute coverage** — Are there any sysfs attributes we missed that would be useful? Any attributes in the guide that have changed names or been deprecated?

2. **Unit conversions** — Can you confirm the conversions are correct (millidegrees, microwatts, millivolts, milliamps)?

3. **Device discovery** — Does the sysfs directory traversal (`tenstorrent!N` → parent → parent for PCI BDF, sibling `hwmon/` for sensors) hold across different kernel versions and card types?

4. **N300 behavior** — We report 1 device per N300 card since the second ASIC is not exposed via sysfs. Is this the expected behavior, or is there a way to discover remote ASICs without pyluwen?

5. **Power readings accuracy** — Are `power1_input` readings from hwmon considered accurate for Tenstorrent cards, or are they estimates?

6. **PyTorch device string** — We used `"tt:N"` (matching tt-metalium convention). Is this the correct device string for current SDK versions?

7. **Anything else** — Any other feedback, corrections, or suggestions are welcome.
