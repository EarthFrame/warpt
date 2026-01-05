# warpt CLI Specification

> Commands: `energyframe`, `upload`, `recommend`

______________________________________________________________________

## Table of Contents

1. [energyframe](#energyframe)
1. [upload](#upload)
1. [recommend](#recommend)
1. [Common Options](#common-options)
1. [Exit Codes](#exit-codes)
1. [Configuration](#configuration)

______________________________________________________________________

## energyframe

Calculate and optionally certify the EnergyFrame efficiency score for the current system.

### Synopsis

```
warpt energyframe [OPTIONS]
```

### Options

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--certify` | `-c` | flag | false | Submit results to API and receive official certificate |
| `--category` | | choice | all | Score category: `inference`, `compute`, `deployment`, `all` |
| `--duration` | `-d` | string | 5m | Benchmark duration per test |
| `--output` | `-o` | path | stdout | Write results to file |
| `--format` | `-f` | choice | text | Output format: `text`, `json`, `yaml` |
| `--quiet` | `-q` | flag | false | Suppress progress output |
| `--verbose` | `-v` | flag | false | Show detailed metrics |

### Examples

#### Basic score (local only)

```bash
warpt energyframe
```

Output:

```
⚡ EnergyFrame Score

Hardware: 2x NVIDIA RTX 4090, AMD EPYC 7313, 256GB RAM

Running benchmarks...
  [████████████████████████████████] 100% Inference tests
  [████████████████████████████████] 100% Compute tests
  [████████████████████████████████] 100% Power measurements

Results:
  Overall Grade: A
  Overall Score: 87.3 / 100

  Category Breakdown:
    Inference (EF²-I):   A+  92.1
    Compute (EF²-C):     A   86.4
    Deployment (EF²-D):  A   85.2

  Key Metrics:
    Peak Efficiency:     1.82 TFLOPS/W
    Idle Power:          42W (4.9% of TDP)
    Tokens/Joule:        24.3 (Llama 3.1 70B Q4)

  Bottlenecks: None detected

To get an official certificate, run:
  warpt energyframe --certify
```

#### Get official certification

```bash
warpt energyframe --certify
```

Output:

```
⚡ EnergyFrame Certification

Hardware: 2x NVIDIA RTX 4090, AMD EPYC 7313, 256GB RAM

Running certification benchmarks...
  [████████████████████████████████] 100% Complete

Submitting to EnergyFrame API...

✓ Certification Complete

  Overall Grade: A
  Overall Score: 87.3 / 100

  Category Breakdown:
    Inference (EF²-I):   A+  92.1
    Compute (EF²-C):     A   86.4
    Deployment (EF²-D):  A   85.2

  Certificate ID: ef2-h-abc123def456
  Certificate URL: https://ef2.dev/cert/ef2-h-abc123def456
  Badge URL: https://ef2.dev/badge/ef2-h-abc123def456.svg
  Valid Until: 2026-12-04

  Carbon Impact:
    Estimated CO₂/hour (inference): 0.34 kg
    Estimated CO₂/hour (idle): 0.02 kg
```

#### Inference-only score

```bash
warpt energyframe --category inference --duration 10m
```

#### JSON output for automation

```bash
warpt energyframe --format json --output energyframe.json
```

Output (energyframe.json):

```json
{
  "version": "1.0",
  "timestamp": "2025-12-04T14:30:00Z",
  "hardware": {
    "accelerators": [
      {"id": "nvidia-rtx-4090", "name": "NVIDIA RTX 4090", "count": 2}
    ],
    "cpu": {"model": "AMD EPYC 7313", "cores": 16, "threads": 32},
    "memory_gb": 256,
    "storage": {"type": "nvme", "capacity_gb": 2000}
  },
  "scores": {
    "overall": {"grade": "A", "score": 87.3},
    "inference": {"grade": "A+", "score": 92.1},
    "compute": {"grade": "A", "score": 86.4},
    "deployment": {"grade": "A", "score": 85.2}
  },
  "metrics": {
    "peak_tflops_fp16": 165.2,
    "peak_efficiency_tflops_per_watt": 1.82,
    "idle_power_watts": 42,
    "load_power_watts": 850,
    "idle_power_ratio": 0.049,
    "tokens_per_second": 45.2,
    "tokens_per_joule": 24.3,
    "memory_bandwidth_gbps": 1008,
    "thermal_throttle_events": 0
  },
  "bottlenecks": [],
  "certification": null
}
```

#### Certified JSON output

```bash
warpt energyframe --certify --format json
```

Adds to JSON:

```json
{
  "certification": {
    "id": "ef2-h-abc123def456",
    "url": "https://ef2.dev/cert/ef2-h-abc123def456",
    "badge_url": "https://ef2.dev/badge/ef2-h-abc123def456.svg",
    "valid_until": "2026-12-04",
    "carbon_estimate": {
      "kg_co2_per_hour_inference": 0.34,
      "kg_co2_per_hour_idle": 0.02
    }
  }
}
```

______________________________________________________________________

## upload

Upload benchmark results, hardware specs, or EnergyFrame data to the warpt API.

### Synopsis

```
warpt upload [OPTIONS] [FILE...]
```

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `FILE` | No | One or more JSON/YAML files to upload. If omitted, reads from stdin. |

### Options

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--type` | `-t` | choice | auto | Data type: `benchmark`, `hardware`, `energyframe`, `auto` |
| `--tag` | | string | | Add a tag to the upload (can be repeated) |
| `--note` | `-n` | string | | Add a note/description |
| `--public` | | flag | false | Make results publicly visible |
| `--quiet` | `-q` | flag | false | Suppress output except errors |
| `--dry-run` | | flag | false | Validate without uploading |

### Examples

#### Upload benchmark results

```bash
warpt benchmark gpu --format json --output results.json
warpt upload results.json
```

Output:

```
Uploading results.json...

✓ Upload Complete

  Type: benchmark
  Result ID: res-789xyz
  Accelerator: nvidia-rtx-4090
  Model: llama-3.1-70b

  Performance:
    Tokens/s: 45.2
    Theoretical %: 94.2%

  Comparison:
    vs Personal Best: +2.3% ✓ New record!
    vs Community Median: +8.1%
    Percentile: 73rd

View at: https://warpt.dev/results/res-789xyz
```

#### Upload from stdin (pipe from benchmark)

```bash
warpt benchmark gpu --format json | warpt upload
```

#### Upload multiple files

```bash
warpt upload benchmark1.json benchmark2.json hardware.json
```

Output:

```
Uploading 3 files...

  benchmark1.json  ✓ res-abc123
  benchmark2.json  ✓ res-def456
  hardware.json    ✓ hw-ghi789

✓ 3 files uploaded successfully
```

#### Upload with tags

```bash
warpt upload results.json --tag "production" --tag "weekly-test"
```

#### Upload with note

```bash
warpt upload results.json --note "After thermal paste replacement"
```

#### Dry run (validate only)

```bash
warpt upload results.json --dry-run
```

Output:

```
Validating results.json...

✓ Valid benchmark result
  Accelerator: nvidia-rtx-4090
  Model: llama-3.1-70b
  Metrics: 12 fields

No errors. Ready to upload.
```

#### JSON output

```bash
warpt upload results.json --format json
```

Output:

```json
{
  "uploads": [
    {
      "file": "results.json",
      "status": "success",
      "type": "benchmark",
      "id": "res-789xyz",
      "url": "https://warpt.dev/results/res-789xyz",
      "comparison": {
        "vs_personal_best_pct": 2.3,
        "is_new_personal_best": true,
        "vs_community_median_pct": 8.1,
        "community_percentile": 73
      }
    }
  ],
  "total": 1,
  "succeeded": 1,
  "failed": 0
}
```

______________________________________________________________________

## recommend

Get model recommendations based on your hardware. Requires prior upload of hardware specs or benchmark results.

### Synopsis

```
warpt recommend [OPTIONS]
```

### Options

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--hardware` | `-h` | string | latest | Hardware ID or "latest" to use most recent upload |
| `--task` | `-t` | choice | inference | Task type: `inference`, `training`, `both` |
| `--max-vram` | | int | auto | Maximum VRAM to consider (GB) |
| `--min-speed` | | float | 0 | Minimum tokens/s required |
| `--quantization` | `-q` | choice | any | Preferred quantization: `fp16`, `int8`, `int4`, `gguf`, `any` |
| `--limit` | `-l` | int | 10 | Number of recommendations |
| `--format` | `-f` | choice | text | Output format: `text`, `json`, `yaml` |

### Examples

#### Basic recommendations

```bash
warpt recommend
```

Output:

```
📊 Model Recommendations

Based on: 2x NVIDIA RTX 4090 (48GB total VRAM)
Task: Inference

Top Recommendations:

 1. Llama 3.1 70B (Q4_K_M)
    ├─ Expected: 45 tok/s
    ├─ VRAM: 44GB (92% utilization)
    ├─ Fit: ★★★★★ Excellent
    └─ Why: Optimal size for your VRAM, great quality/speed balance

 2. Mixtral 8x7B (Q5_K_M)
    ├─ Expected: 62 tok/s
    ├─ VRAM: 38GB (79% utilization)
    ├─ Fit: ★★★★★ Excellent
    └─ Why: Fast MoE architecture, excellent for varied tasks

 3. Qwen2 72B (Q4_K_M)
    ├─ Expected: 41 tok/s
    ├─ VRAM: 46GB (96% utilization)
    ├─ Fit: ★★★★☆ Good
    └─ Why: Strong multilingual, tight VRAM fit

 4. Llama 3.1 70B (Q5_K_M)
    ├─ Expected: 38 tok/s
    ├─ VRAM: 51GB ⚠️ Requires offloading
    ├─ Fit: ★★★☆☆ Possible
    └─ Why: Higher quality but needs CPU offload

 5. DeepSeek-V2 (Q4_K_M)
    ├─ Expected: 58 tok/s
    ├─ VRAM: 42GB (88% utilization)
    ├─ Fit: ★★★★★ Excellent
    └─ Why: Efficient MoE, great for coding tasks

Models that DON'T fit:
  • Llama 3.1 405B - Requires 220GB+ VRAM
  • Mixtral 8x22B - Requires 85GB+ VRAM

Run `warpt benchmark gpu --model <name>` to test actual performance.
```

#### Filter by minimum speed

```bash
warpt recommend --min-speed 50
```

#### Filter by quantization preference

```bash
warpt recommend --quantization fp16
```

Output:

```
📊 Model Recommendations (FP16 only)

Based on: 2x NVIDIA RTX 4090 (48GB total VRAM)

Top Recommendations:

 1. Llama 3.1 8B (FP16)
    ├─ Expected: 85 tok/s
    ├─ VRAM: 16GB (33% utilization)
    ├─ Fit: ★★★★☆ Good (underutilized)
    └─ Why: Full precision, very fast, but hardware underused

 2. Mistral 7B (FP16)
    ├─ Expected: 92 tok/s
    ├─ VRAM: 14GB (29% utilization)
    ├─ Fit: ★★★★☆ Good (underutilized)
    └─ Why: Full precision, excellent speed

Note: FP16 models are limited by VRAM. Consider quantized
models to run larger architectures on your hardware.
```

#### Use specific hardware profile

```bash
warpt recommend --hardware hw-abc123
```

#### JSON output

```bash
warpt recommend --format json --limit 3
```

Output:

```json
{
  "hardware": {
    "id": "hw-latest",
    "accelerators": [
      {"id": "nvidia-rtx-4090", "count": 2, "vram_gb": 24}
    ],
    "total_vram_gb": 48
  },
  "task": "inference",
  "recommendations": [
    {
      "rank": 1,
      "model_id": "meta-llama-3.1-70b",
      "model_name": "Llama 3.1 70B",
      "quantization": "Q4_K_M",
      "expected_tokens_per_second": 45.2,
      "vram_required_gb": 44,
      "vram_utilization_pct": 92,
      "fit_score": 5,
      "fit_label": "Excellent",
      "reasoning": "Optimal size for your VRAM, great quality/speed balance"
    },
    {
      "rank": 2,
      "model_id": "mistralai-mixtral-8x7b",
      "model_name": "Mixtral 8x7B",
      "quantization": "Q5_K_M",
      "expected_tokens_per_second": 62.1,
      "vram_required_gb": 38,
      "vram_utilization_pct": 79,
      "fit_score": 5,
      "fit_label": "Excellent",
      "reasoning": "Fast MoE architecture, excellent for varied tasks"
    },
    {
      "rank": 3,
      "model_id": "qwen-qwen2-72b",
      "model_name": "Qwen2 72B",
      "quantization": "Q4_K_M",
      "expected_tokens_per_second": 41.0,
      "vram_required_gb": 46,
      "vram_utilization_pct": 96,
      "fit_score": 4,
      "fit_label": "Good",
      "reasoning": "Strong multilingual, tight VRAM fit"
    }
  ],
  "excluded": [
    {
      "model_id": "meta-llama-3.1-405b",
      "reason": "Requires 220GB+ VRAM"
    }
  ]
}
```

______________________________________________________________________

## Common Options

These options are available on all commands:

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--config` | `-c` | path | ~/.warpt/config.toml | Config file path |
| `--api-url` | | string | https://api.warpt.dev | API base URL |
| `--api-key` | | string | from config | API key for authenticated requests |
| `--format` | `-f` | choice | text | Output format: `text`, `json`, `yaml` |
| `--no-color` | | flag | false | Disable colored output |
| `--debug` | | flag | false | Enable debug logging |
| `--help` | `-h` | flag | | Show help message |
| `--version` | `-V` | flag | | Show version |

______________________________________________________________________

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Invalid arguments |
| 3 | Authentication required |
| 4 | API error |
| 5 | Hardware detection failed |
| 6 | Benchmark failed |
| 7 | Upload failed |
| 8 | Network error |

______________________________________________________________________

## Configuration

### Config File Location

```
~/.warpt/config.toml
```

### Config File Format

```toml
[api]
url = "https://api.warpt.dev"
key = "warpt_sk_xxxxxxxxxxxxx"

[defaults]
format = "text"
color = true

[energyframe]
default_duration = "5m"
auto_upload = false

[upload]
default_public = false
default_tags = ["automated"]
```

### Environment Variables

| Variable | Overrides |
|----------|-----------|
| `WARPT_API_URL` | api.url |
| `WARPT_API_KEY` | api.key |
| `WARPT_CONFIG` | config file path |
| `NO_COLOR` | defaults.color |

______________________________________________________________________

## Authentication

Commands that interact with the API require authentication:

| Command | Auth Required |
|---------|---------------|
| `energyframe` (local) | No |
| `energyframe --certify` | Yes |
| `upload` | Yes |
| `recommend` | Yes |

### Login

```bash
warpt login
# Opens browser for OAuth flow

warpt login --token warpt_sk_xxxxx
# Direct token authentication
```

### Check Auth Status

```bash
warpt auth status
```

Output:

```
✓ Authenticated

  User: user@example.com
  Tier: Pro
  API Calls Today: 23 / 100
  Expires: 2026-01-15
```

______________________________________________________________________

*CLI Specification v1.0 — warpt by EarthFrame*
