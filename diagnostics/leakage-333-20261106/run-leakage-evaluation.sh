export PYTHONDONTWRITEBYTECODE=1
export TMPDIR=/mnt/e/Company/_codex_research_20261106/eda
export XDG_CACHE_HOME=/mnt/e/Company/.temp-tools/contribution-runtime/cache
export MPLCONFIGDIR=/mnt/e/Company/.temp-tools/contribution-runtime/matplotlib
export NUMBA_CACHE_DIR=/mnt/e/Company/.temp-tools/contribution-runtime/numba
export PYTHONPYCACHEPREFIX=/mnt/e/Company/.temp-tools/contribution-runtime/pycache
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
cd /mnt/e/Company/_codex_research_20261106/eda
/root/openram-venv/v/bin/python -u evaluate-leakage.py > leakage-comparison.log 2>&1
