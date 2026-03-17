#!/bin/bash
set -e

if [ -n "$ASCEND_INSTALL_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_INSTALL_PATH
elif [ -n "$ASCEND_HOME_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_HOME_PATH
else
    if [ -d "$HOME/Ascend/ascend-toolkit/latest" ]; then
        _ASCEND_INSTALL_PATH=$HOME/Ascend/ascend-toolkit/latest
    else
        _ASCEND_INSTALL_PATH=/usr/local/Ascend/ascend-toolkit/latest
    fi
fi

set +e
source $_ASCEND_INSTALL_PATH/bin/setenv.bash
set -e

USER_ABI_VERSION_RAW=$(python3 -c "import torch; print(1 if torch.compiled_with_cxx11_abi() else 0)")
if [ $? -ne 0 ]; then
    echo "Error: Failed to retrieve PyTorch ABI version!"
    echo "Possible reasons: PyTorch is not installed, or the version is too old (missing the _GLIBCXX_USE_CXX11_ABI attribute)."
    exit 1
fi
export USER_ABI_VERSION=$(echo "$USER_ABI_VERSION_RAW" | tr -d '[:space:]')


rm -rf build
mkdir -p build
cmake -B build ../csrc
cmake --build build -j