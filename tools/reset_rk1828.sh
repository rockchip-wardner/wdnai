#!/bin/sh
# Reset the RK1828 NPU driver state. Clears the wedged-context state
# that builds up after a series of failed inference cycles (RPC
# timeouts, MODEL_SETUP fail loops).
#
# Two layers of reset, in order:
#   1. `sudo rknn-smi reset -t hw -d N` — **hardware** reset of the
#      PCIe RK1828 card at index N. Reflashes the on-card firmware.
#      This is the only step that actually clears the wedged-context
#      state in the kernel-side driver.
#   2. `sudo rknn3_startup restart` — restarts rknn_server + the
#      per-device transfer_proxy daemons (userspace half).
#
# Run the hardware reset 4 times back-to-back because a single reset
# is not always sufficient to clear a deeply wedged state. (This
# matches the operator recipe that has been observed to recover
# qwen3-vl-4b after a wedge storm.)
#
# IMPORTANT — what this script does NOT do:
#   * It does NOT unload/reload the kernel-side rknn driver module.
#     On the target board the driver is built-in (not a module), so
#     `rmmod rknn` is not available. For full kernel-state reset you
#     still need `sudo reboot`.
#   * It does NOT kill the wdnwebui uvicorn / run.py process. The
#     supervisor will reconnect to the daemon automatically.
#
# Usage:
#   sudo tools/reset_rk1828.sh              # full reset (default NPU 0)
#   sudo tools/reset_rk1828.sh --npu 1     # reset NPU 1 instead
#   sudo tools/reset_rk1828.sh --keep-web  # also keep wdnai workers
#                                            # alive (skip the worker
#                                            # SIGTERM pass)
#
# Exit codes:
#   0   reset completed cleanly (transfer_proxy back up)
#   1   reset failed (transfer_proxy did not come back within timeout)

set -e

LOG() { printf "[reset_rk1828] %s\n" "$*"; }

NPU=0
KEEP_WEB=0
for arg in "$@"; do
    case "$arg" in
        --npu) shift; NPU="$1" ;;
        --npu=*) NPU="${arg#--npu=}" ;;
        --keep-web) KEEP_WEB=1 ;;
        -h|--help)
            sed -n '2,/^$/s/^# \{0,1\}//p' "$0" | head -50
            exit 0
            ;;
        *) LOG "unknown arg: $arg"; exit 2 ;;
    esac
    shift 2>/dev/null || true
done

# ---------------------------------------------------------------------------
# 0. Sanity check — confirm rknn tooling is present.
# ---------------------------------------------------------------------------
if [ ! -x /usr/bin/rknn-smi ]; then
    LOG "no /usr/bin/rknn-smi — is this a Rockchip target? aborting"
    exit 1
fi
if [ ! -x /usr/bin/rknn3_startup ]; then
    LOG "no /usr/bin/rknn3_startup — aborting"
    exit 1
fi

LOG "target NPU: ${NPU}"

# ---------------------------------------------------------------------------
# 1. Stop the wdnai compute_workers for qwen3-vl-4b (they hold the NPU).
#    The supervisor (uvicorn / run.py) will respawn them when the daemon
#    comes back; this just guarantees the NPU is free during the reset.
# ---------------------------------------------------------------------------
if [ "$KEEP_WEB" -eq 0 ]; then
    LOG "stopping compute_worker subprocess(es) for qwen3-vl-4b"
    pids=$(pgrep -f "compute_worker.py --model qwen3-vl-4b" || true)
    if [ -n "$pids" ]; then
        kill -TERM $pids 2>/dev/null || true
        sleep 2
        survivors=$(pgrep -f "compute_worker.py --model qwen3-vl-4b" || true)
        if [ -n "$survivors" ]; then
            LOG "  SIGKILL stubborn survivor(s): $survivors"
            kill -KILL $survivors 2>/dev/null || true
        fi
    else
        LOG "  none running"
    fi
fi

# ---------------------------------------------------------------------------
# 2. Hardware reset (4× back-to-back to clear wedged state).
# ---------------------------------------------------------------------------
LOG "hardware reset of NPU ${NPU} (4x)"
for i in 1 2 3 4; do
    if ! sudo rknn-smi reset -t hw -d "${NPU}" >/dev/null 2>&1; then
        LOG "  WARN: rknn-smi reset iteration ${i} returned non-zero (continuing)"
    fi
done

# ---------------------------------------------------------------------------
# 3. Restart userspace rknn daemons.
# ---------------------------------------------------------------------------
LOG "restarting rknn_server + transfer_proxy"
if ! sudo rknn3_startup restart >/dev/null 2>&1; then
    LOG "  WARN: rknn3_startup restart returned non-zero (continuing)"
fi

# ---------------------------------------------------------------------------
# 4. Wait for transfer_proxies to come back.
# ---------------------------------------------------------------------------
LOG "waiting for transfer_proxy to be ready"
for i in $(seq 1 30); do
    n=$(rknn-smi -L 2>/dev/null | grep -c -E '^[[:space:]]*[0-9]+[[:space:]]+Online' || true)
    if [ "$n" -ge 2 ]; then
        LOG "  rknn-smi ready after ${i}s ($n devices online)"
        break
    fi
    sleep 1
done

# ---------------------------------------------------------------------------
# 5. Final snapshot.
# ---------------------------------------------------------------------------
LOG "reset complete. rknn-smi snapshot:"
rknn-smi 2>&1 | head -20 || true
echo
LOG "next steps:"
echo "  1. open http://<host>:8000/  -> 算力配置 page"
echo "  2. click 启用 for qwen3-vl-4b, pick an NPU, wait ~30s"
echo "  3. open 音视频描述 page, the #vd-detect-size dropdown should show"
echo "     [384× 768] (or [384] if 768x768 weights are not present)"
exit 0