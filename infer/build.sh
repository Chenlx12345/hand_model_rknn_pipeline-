#!/bin/sh
# 一键交叉编译：使用主仓库 cmake/toolchain-rk3588.cmake
# 产物：build/unit_bench_e2e_test  (aarch64 ELF)
set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
TOOLCHAIN="$REPO_ROOT/cmake/toolchain-rk3588.cmake"
BUILD_DIR="$SCRIPT_DIR/build"

if [ ! -f "$TOOLCHAIN" ]; then
    echo "FAIL: toolchain not found: $TOOLCHAIN" >&2
    exit 1
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
cmake -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" \
      -DCMAKE_BUILD_TYPE=Release \
      "$SCRIPT_DIR"
cmake --build . -j"$(nproc)"

echo
echo "PASS: $BUILD_DIR/unit_bench_e2e_test"
file "$BUILD_DIR/unit_bench_e2e_test" || true
