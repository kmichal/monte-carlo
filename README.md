# monte-carlo

Simple command-line utility to run Monte Carlo simulations for equities.

## Features
- Calibrate GBM parameters (drift, volatility) from historical close prices
- Run single-threaded or parallel Monte Carlo simulations
- Fetch historical market data (from Tradier) via libcurl and parse JSON with nlohmann::json

## Requirements
- C++23 compiler with module support
- CMake (>= 3.31)
- Ninja (recommended generator)
- System libraries: libcurl, nlohmann_json
- Access to Tradier API

## Build
From the repository root:

```bash
cmake -S . -B build -G Ninja
cmake --build ./build/
```

## Usage
Run the built executable and inspect `--help` for flags:

```bash
./build/mc --help
# Example:
./build/mc --tradier-token XYXYXYXYXYXY --symbol BE --lookback-days 100 --days-forward 255 --simulations 10000
```

## Key Files
- [main.cpp](main.cpp) — program entry and CLI handling
- [modules/EquitySim.cpp](modules/EquitySim.cpp) — `EquityEngine`, `SimResult`, `run`, `run_parallel`
- [modules/MarketData.cpp](modules/MarketData.cpp) — HTTP + JSON data retrieval and parsing
- [CMakeLists.txt](CMakeLists.txt) — build configuration and dependency discovery

