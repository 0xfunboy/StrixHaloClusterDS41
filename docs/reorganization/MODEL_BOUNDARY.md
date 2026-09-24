# StrixHaloClusterDS41

Model-family identifier: `deepseek41-flash`. Canonical repository entry: `model-profile.json`. Default `config.json` is now model-specific. Engine reference: `runtime/ds41/presets/dspark-k2-gfx1151.json`.

K2 resident reference; E1 comparator stays a separate model profile. Existing model release files, runtimes, weights, closed campaign protocols and raw evidence are unchanged. Shared API/frontend/ownership source is governed by `common-core.lock.json`. No service was installed/restarted and no model switch was performed during this reorganization.

Both legacy lifecycle API paths remain supported with authenticated explicit confirmation. A future multi-model frontend must coordinate the single shared resource group; separate repositories do not make the two-node resident coexist with a local Qwen instance.
