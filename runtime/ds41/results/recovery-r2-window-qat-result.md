# DS41 RECOVERY R2 — window-QAT differential

Status: **MATERIAL_COMPONENT_EFFECT**.

- rank0 chunk0 rows895..1022: row rel-L2=0.0178698, attention rel-L2=0.00773809, max-abs=0.03125, outside=0/16384, equivalent=True
- rank0 chunk1 rows1460..1587: row rel-L2=0.0175235, attention rel-L2=0.00665825, max-abs=0.03125, outside=0/16384, equivalent=True
- rank1 chunk0 rows895..1022: row rel-L2=0.0178698, attention rel-L2=0.00595624, max-abs=0.0390625, outside=0/16384, equivalent=True
- rank1 chunk1 rows1460..1587: row rel-L2=0.0175235, attention rel-L2=0.0106077, max-abs=0.0703125, outside=3/16384, equivalent=False

One bounded full-model localization window is warranted for a window-QAT semantic patch; this component result is not itself a correctness result.

No claim that upstream QAT is end-to-end correct is made here.
