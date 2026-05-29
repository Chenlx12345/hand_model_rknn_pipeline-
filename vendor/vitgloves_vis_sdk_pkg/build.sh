#!/usr/bin/env bash
# 集成方一键编译:.a + algo_input 已存在的前提下,build/demo_integrator 出来即可。
set -e
SD="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$SD/examples/build"
cd "$SD/examples/build"
cmake .. && make -j$(nproc)
echo
echo "✓ binary: $SD/examples/build/demo_integrator"
echo "  run   : ./demo_integrator [output_path]"
