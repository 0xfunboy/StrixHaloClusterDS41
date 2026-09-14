# Experimental DSpark K=2 preset — gfx1151

This preset is **experimentally qualified** by attempt041/042. It does **not** replace the production M1 default and does not remove the existing qualified K1 preset.

## Reproduce the qualified K2 offline arm

```bash
cd /home/funboy/StrixHaloClusterDS41
ATTEMPT=attempt-dspark-k2-repro
mkdir -p reports/DS41-Q2-001/$ATTEMPT
cp reports/DS41-Q2-001/attempt041-dspark-k2/prompt-tokens.json \
   reports/DS41-Q2-001/$ATTEMPT/prompt-tokens.json
EPOCH=$(date +%s%N)
DS41_RUN_MODE=offline DS41_ATTEMPT_NAME=$ATTEMPT DS41_RUNTIME_MAX_SEC=5400 \
  runtime/ds41/pair.sh start "$EPOCH" 18210
systemd-run --user --unit="ds41-$ATTEMPT-supervisor" --collect \
  --property=RuntimeMaxSec=5500 --property=KillMode=control-group --property=Restart=no \
  /usr/bin/bash /home/funboy/StrixHaloClusterDS41/scripts/supervise-ds41-offline-attempt.sh \
  "$ATTEMPT" "$EPOCH" 5450
```

The frozen fixture sets `dspark_real.arm=dspark` and `num_speculative_tokens=2`. The runner accepts only K=1 or K=2, keeps all three trained MTP stages, target layers 37/38/39, TP2/EP2, adaptive verification OFF, the existing Safetensors sidecar and the qualified gfx1151 BF16 expert fallback. It does not install an oracle/replay proposer.

## Roll back K2 -> K1

Use the existing preset:

```text
runtime/ds41/presets/dspark-k1-gfx1151.json
```

or reproduce the contemporary control with:

```text
reports/DS41-Q2-001/attempt042-dspark-k1-control/prompt-tokens.json
```

The only draft-width change is `num_speculative_tokens: 2 -> 1`; keep the same sidecar, loader, quantization, target switches and DS41 rowwise target controls.

## Roll back K1 -> M1

Use `speculative_config=None` and unset `DS41_DSPARK_MXFP4_BF16`, `DS41_NATIVE_HIP_MOE_ROWWISE`, `DS41_MHC_ROWWISE_BLOCK`, and `DS41_REAL_DSPARK_TELEMETRY`, while retaining the promoted target switches. This is the same rollback already documented by `dspark-k1-gfx1151`.

## Qualified result

Three speed128 repetitions: K1 mean **16.83498 tok/s**, K2 mean **19.43182 tok/s**, **+15.4253%**. The frozen validator PASSes rank coherence, exact K1/K2 token/text/finish/stop equality across the full panel, arithmetic/JSON/reasoning/text, and independent coding 9/9 in both arms.

K2 full-width speed acceptance histogram for 0/1/2 accepted drafts is **24 / 39 / 93** over 156 verification calls. Proposer GPU-stream cost rises from **12.003 ms/call K1** to **13.958 ms/call K2**, while proposal calls fall from 213 to 159 and total proposer GPU span falls from 2556.619 ms to 2219.309 ms.

No API/frontend or production default is modified by this preset. K3/K4 are outside this qualification.
