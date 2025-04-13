# HashBreaker
[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/DGIorio)

A GPU-accelerated (CUDA) tool designed to brute-force custom/non-standard CRC32 hashes. It generates candidate strings with optional constrains (prefix, suffix, and contains), reads multiple target hashes from a file, and finds all matches within configured variable string length. Key parameters are managed via a configuration file.

## Requirements
* **NVIDIA GPU:** Compute Capability 3.0 or higher (tested on a RTX 4060).
* **CUDA Toolkit:** Version 12.x recommended (developed with 12.8). Download from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit).
* **C++ Compiler:** Supporting C++20 standard (developed with Visual Studio 2022).
* **Operating System:** Developed and tested on Windows 11. May require adjustments for other operating systems.

## Usage
Command-line usage: `.\HashBreaker.exe <config> <hashes>`

Input data:  
 `<config>`     Configuration file with key parameters  
 `<hashes>`     Input file with target hashes list

## Features
* CUDA Acceleration: Uses GPU power for significantly faster brute-forcing over CPU-based methods.
* String Constraints: Supports optional `PREFIX`, `SUFFIX`, and `CONTAINS` constraints for targeted searches.
* Finds All Matches: Continues searching and reports all matching strings found for each target hash within the configured limits.
