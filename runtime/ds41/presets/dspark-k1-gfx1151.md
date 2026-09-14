# Experimental DSpark K=1 preset — gfx1151

This preset is **qualified experimentally**, not promoted as the production default. It describes the exact B039 configuration that passed the A040 comparison.

## Reproduce the qualified offline arm

```bash
cd /home/funboy/StrixHaloClusterDS41
ATTEMPT=attempt-dspark-k1-repro
mkdir -p reports/DS41-Q2-001/$ATTEMPT
cp reports/DS41-Q2-001/attempt039-dspark-k1/prompt-tokens.json reports/DS41-Q2-001/$ATTEMPT/prompt-tokens.json
EPOCH=$(date +%s%N)
DS41_RUN_MODE=offline DS41_ATTEMPT_NAME=$ATTEMPT DS41_RUNTIME_MAX_SEC=5400 \
  runtime/ds41/pair.sh start "$EPOCH" 18210
systemd-run --user --unit="ds41-$ATTEMPT-supervisor" --collect \
  --property=RuntimeMaxSec=5500 --property=KillMode=control-group --property=Restart=no \
  /usr/bin/bash /home/funboy/StrixHaloClusterDS41/scripts/supervise-ds41-offline-attempt.sh \
  "$ATTEMPT" "$EPOCH" 5450
```

The frozen `dspark_real.arm=dspark` fixture applies the preset before model construction. No oracle/replay proposer is installed.

## Roll back to M1

Use `dspark_real.arm=m1` / `speculative_config=None` and unset `DS41_DSPARK_MXFP4_BF16`, `DS41_NATIVE_HIP_MOE_ROWWISE`, `DS41_MHC_ROWWISE_BLOCK`, and `DS41_REAL_DSPARK_TELEMETRY`. Keep the promoted target switches enabled. `attempt040-m1-control/prompt-tokens.json` is the frozen rollback/control fixture.

No API/frontend production default is modified by this preset.
