# localscore-bench

A reimplementation of [LocalScore](https://localscore.ai) using `llama-bench` as the underlying benchmarking engine.

## Overview

This tool wraps `llama-bench` to provide the same benchmarking experience as the original localscore tool, including:

- Running 9 predefined benchmark tests covering various prompt/generation ratios
- Calculating the LocalScore performance metric
- Displaying results in a formatted table
- Optional submission of results to localscore.ai

## Requirements

- Python 3.8+
- `llama-bench` binary (from llama.cpp)
- A GGUF model file

## Installation

1. Clone or copy this directory
2. Ensure `llama-bench` is available (either in PATH or specify with `--llama-bench`)

No additional Python dependencies are required - the tool uses only the standard library.

## Usage

```bash
# Basic usage
python main.py -m /path/to/model.gguf

# Specify llama-bench location
python main.py -m model.gguf --llama-bench /path/to/llama-bench

# Run on CPU only
python main.py -m model.gguf -c

# Run extended tests (4 repetitions)
python main.py -m model.gguf -e

# Submit results automatically
python main.py -m model.gguf -y

# Quick benchmark (3 tests instead of 9)
python main.py -m model.gguf --quick

# JSON output
python main.py -m model.gguf -o json
```

## Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `-m, --model` | Path to GGUF model file (required) | - |
| `--llama-bench` | Path to llama-bench binary | auto-detect |
| `-c, --cpu` | Disable GPU acceleration | off |
| `-g, --gpu` | GPU backend: auto\|amd\|apple\|nvidia\|disabled | auto |
| `-i, --gpu-index` | Select GPU by index | 0 |
| `-t, --threads` | Number of CPU threads | auto |
| `-o, --output` | Output format: console\|json\|csv | console |
| `-v, --verbose` | Enable verbose output | off |
| `--plaintext` | Plaintext output (no ASCII art) | off |
| `-y, --send-results` | Submit results without asking | off |
| `-n, --no-send-results` | Never submit results | off |
| `-e, --extended` | Run 4 repetitions | off |
| `--long` | Run 16 repetitions | off |
| `--reps N` | Custom number of repetitions | 1 |
| `--quick` | Run only 3 representative tests | off |

## Benchmark Tests

The tool runs 9 predefined tests matching the original localscore:

| Test | Prompt | Gen | Ratio | Use Case |
|------|--------|-----|-------|----------|
| pp1024+tg16 | 1024 | 16 | 64:1 | Title generation |
| pp4096+tg256 | 4096 | 256 | 16:1 | Content summarization |
| pp2048+tg256 | 2048 | 256 | 8:1 | Code review/fix |
| pp2048+tg768 | 2048 | 768 | 3:1 | Standard code chat |
| pp1024+tg1024 | 1024 | 1024 | 1:1 | Code back-and-forth |
| pp1280+tg3072 | 1280 | 3072 | 1:2.4 | Reasoning over code |
| pp384+tg1152 | 384 | 1152 | 1:3 | Code gen with interaction |
| pp64+tg1024 | 64 | 1024 | 1:16 | Code gen/ideation |
| pp16+tg1536 | 16 | 1536 | 1:96 | QA, Storytelling, Reasoning |

## Score Calculation

The LocalScore is calculated as a geometric mean:

```
score = 10 * (avg_prompt_tps * avg_gen_tps * (1000 / avg_ttft_ms))^(1/3)
```

Where:
- `avg_prompt_tps` - Average prompt processing tokens per second
- `avg_gen_tps` - Average token generation tokens per second
- `avg_ttft_ms` - Average time to first token in milliseconds

### Score Guidelines

- **1000+**: Excellent
- **500+**: Very Good
- **250+**: Good
- **100+**: Acceptable
- **<100**: Poor

## Output Formats

### Console (default)
```
+------------------------------------------------------------------------------+
|                    NVIDIA RTX PRO 6000 - 48.0 GiB                            |
|                      Llama 3.2 3B - Q4_K_M                                   |
+------------------------------------------------------------------------------+
|          test |    run |     avg time |           tokens |     pp t/s | ...
| ------------- | ------ | ------------ | ---------------- | ---------- | ...
|   pp1024+tg16 |   1/1  |     32.45 ms |       1040 / ... |    2156.32 | ...
...

     ██╗ ██████╗  ██████╗ ██████╗
    ███║██╔═████╗██╔═████╗╚════██╗
    ╚██║██║██╔██║██║██╔██║ █████╔╝
     ██║████╔╝██║████╔╝██║██╔═══╝
     ██║╚██████╔╝╚██████╔╝███████╗
     ╚═╝ ╚═════╝  ╚═════╝ ╚══════╝

Token Generation:        156.78 tok/s
Prompt Processing:       2156.32 tok/s
Time to First Token:     45.23 ms
```

### JSON
```json
{
  "runtime_info": { ... },
  "system_info": { ... },
  "accelerator_info": { ... },
  "results": [ ... ],
  "results_summary": {
    "avg_prompt_tps": 2156.32,
    "avg_gen_tps": 156.78,
    "avg_ttft_ms": 45.23,
    "performance_score": 1002.5
  }
}
```

### CSV
```csv
test,n_prompt,n_gen,prompt_tps,gen_tps,ttft_ms
pp1024+tg16,1024,16,2156.32,156.78,45.23
...
```

## Result Submission

Results can be submitted to [localscore.ai](https://localscore.ai) for comparison with other hardware configurations.

- Use `-y` to submit without confirmation
- Use `-n` to never submit
- By default, you'll be asked for confirmation

Submitted data includes:
- System information (CPU, RAM, OS)
- GPU/Accelerator information
- Benchmark results and score

## Differences from Original localscore

| Feature | Original localscore | localscore-bench |
|---------|--------------------|--------------------|
| Backend | Built-in llama.cpp | External llama-bench |
| Power monitoring | Yes | No |
| Live progress updates | Yes | Basic |
| TTFT measurement | Direct | Calculated |
| GPU auto-selection | Interactive | CLI flag |

## License

MIT License
