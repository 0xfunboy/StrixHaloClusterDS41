# DS41 RECOVERY R2 decision

Status: **R2_NO_ISOLATED_LOCAL_FIX_PROCEED_R4**.

The same-input window-QAT differential is material on rank1/chunk1, but the local cache ABI cannot preserve the tested full512/block32 representation without changing writer/page/gather contracts. The existing decomposed path ends in the same hybrid `fp8_ds_mla` cache and would double-quantize a QDQ input. Because compressed/indexer QAT and Engram storage also differ, that would not be a causal single-variable fix.

No full-model localization window was consumed. Proceed to R4 compatibility/build gates; no model acquisition until all mandate conditions pass.
