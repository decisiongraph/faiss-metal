{ pkgs, lib, config, inputs, ... }:

{
  # Only provide dependencies not available in macOS SDK.
  # Metal frameworks, ObjC++ compiler, and Accelerate come from Xcode.
  packages = [
    pkgs.git
    pkgs.cmake
    pkgs.faiss
    pkgs.llvmPackages.openmp
  ];

  # Do NOT enable languages.cplusplus — it pulls in nix apple-sdk which
  # conflicts with the real macOS SDK needed for Metal/MPS headers.

  # Nix overrides DEVELOPER_DIR to its stripped apple-sdk which lacks Metal tools.
  # We must restore it to the real Xcode path for Metal shader compilation.
  env.DEVELOPER_DIR = lib.mkForce "/Applications/Xcode.app/Contents/Developer";

  enterShell = ''
    # Force real Xcode DEVELOPER_DIR (nix apple-sdk lacks Metal tools)
    export DEVELOPER_DIR="/Applications/Xcode.app/Contents/Developer"

    # Unset nix SDKROOT — it points to nix apple-sdk which lacks libc++ and
    # Metal/MPS headers needed by the system compiler and linker.
    unset SDKROOT
    unset NIX_CFLAGS_COMPILE
    unset NIX_LDFLAGS

    export CC=/usr/bin/clang
    export CXX=/usr/bin/clang++

    echo "faiss-metal dev environment"
    echo "  build  - configure & build"
    echo "  test   - build & run tests"
    echo "  bench  - build & run benchmarks"
  '';

  scripts.build.exec = ''
    export DEVELOPER_DIR="/Applications/Xcode.app/Contents/Developer"
    unset SDKROOT
    unset NIX_CFLAGS_COMPILE
    unset NIX_LDFLAGS
    CC=/usr/bin/clang CXX=/usr/bin/clang++ \
    cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH="${pkgs.faiss};${pkgs.llvmPackages.openmp}" \
      -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I${pkgs.llvmPackages.openmp.dev}/include" \
      -DOpenMP_CXX_LIB_NAMES="omp" \
      -DOpenMP_omp_LIBRARY="${pkgs.llvmPackages.openmp}/lib/libomp.dylib"
    cmake --build build -j
  '';

  scripts.test.exec = ''
    export DEVELOPER_DIR="/Applications/Xcode.app/Contents/Developer"
    unset SDKROOT
    unset NIX_CFLAGS_COMPILE
    unset NIX_LDFLAGS
    CC=/usr/bin/clang CXX=/usr/bin/clang++ \
    cmake -B build -DCMAKE_BUILD_TYPE=Debug -DFAISS_METAL_BUILD_TESTS=ON \
      -DCMAKE_PREFIX_PATH="${pkgs.faiss};${pkgs.llvmPackages.openmp}" \
      -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I${pkgs.llvmPackages.openmp.dev}/include" \
      -DOpenMP_CXX_LIB_NAMES="omp" \
      -DOpenMP_omp_LIBRARY="${pkgs.llvmPackages.openmp}/lib/libomp.dylib"
    cmake --build build -j
    cd build && ctest --output-on-failure
  '';

  scripts.bench.exec = ''
    export DEVELOPER_DIR="/Applications/Xcode.app/Contents/Developer"
    unset SDKROOT
    unset NIX_CFLAGS_COMPILE
    unset NIX_LDFLAGS
    CC=/usr/bin/clang CXX=/usr/bin/clang++ \
    cmake -B build -DCMAKE_BUILD_TYPE=Release -DFAISS_METAL_BUILD_TESTS=ON \
      -DCMAKE_PREFIX_PATH="${pkgs.faiss};${pkgs.llvmPackages.openmp}" \
      -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I${pkgs.llvmPackages.openmp.dev}/include" \
      -DOpenMP_CXX_LIB_NAMES="omp" \
      -DOpenMP_omp_LIBRARY="${pkgs.llvmPackages.openmp}/lib/libomp.dylib"
    cmake --build build -j
    ./build/bench_metal_flat
  '';
}
