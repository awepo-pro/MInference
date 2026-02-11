# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

.PHONY: install style test

PYTHON := python
CHECK_DIRS := minference tests experiments examples
EXELUDE_DIRS := minference/ops
FORCE_EXELUDE_DIRS := minference/modules/minference_forward.py

install:
	@${PYTHON} setup.py bdist_wheel
	@${PYTHON} -m pip install dist/minference*

style:
	black $(CHECK_DIRS) --extend-exclude ${EXELUDE_DIRS} --force-exclude ${FORCE_EXELUDE_DIRS}
	isort -rc $(CHECK_DIRS)

# test:
# 	@${PYTHON} -m pytest -n 1 --dist=loadfile -s -v ./tests/

test:
	@VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_V1=0 ${PYTHON} test.py 


# benchmark:
# 	@VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_V1=0 ${PYTHON} benchmark2.py --context_window 24_000

benchmark:
	@VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_V1=0 ${PYTHON} benchmark4.py

run:
	@VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_V1=0 ${PYTHON} main.py 
