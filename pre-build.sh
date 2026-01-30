export CPATH=$CUDA_HOME/include:$CPATH

# cicc
export PATH="${CONDA_PREFIX}/nvvm/bin:${PATH}"

# nvcc
export PATH="${CONDA_PREFIX}/targets/x86_64-linux:$PATH"
export CUDA_HOME="${CONDA_PREFIX}/targets/x86_64-linux"
export CUDA_PATH="$CUDA_HOME"

# Correct path for Conda-based CUDA headers
export CUDA_INCLUDE_DIRS="$CUDA_HOME/include"

# Fix paths to use the actual Python 3.10 version found in your environment
export CPATH="${CONDA_PREFIX}/include/python3.10:$CPATH"

# Help the linker find libraries
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export TORCH_CUDA_ARCH_LIST="8.6"

export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

# echo "PATH: ${PATH}"

echo "CUDA_HOME: $CUDA_HOME"
echo "nvcc: $(which nvcc)"

echo "cicc: $(which cicc)"

uv add --dev minference
