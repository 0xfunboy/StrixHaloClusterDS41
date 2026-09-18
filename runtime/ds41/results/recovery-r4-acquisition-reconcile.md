# DS41 Recovery R4 acquisition reconcile

NODE02 already has the exact Q2 payload and its existing acquisition registry records full size+SHA verification. The addendum pin d12db970... and acquired pin dd8a266f... resolve via the HF tree API to the same file oid/LFS oid/size, so no second 340.6 GiB Internet transfer is justified.

NODE01 mirror is BLOCKED_MARGIN: 372.8501 GiB user-available before mirror, projected 32.2527 GiB after the 340.5974 GiB payload, versus the mandated >=50 GiB residual. Deficit: 17.7473 GiB. No alternate large filesystem exists. No deletion or mirror was performed. K2 serving remains READY/untouched.
