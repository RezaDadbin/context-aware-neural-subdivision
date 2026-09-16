# Local Source Meshes

Source meshes are not distributed in this repository. Place legally obtained,
closed triangular meshes here before running the three-object reproduction
scripts:

```text
01_bimba_head.obj
02_fat_dragon.obj
03_gear16.obj
```

The default generator creates randomized `500 -> 2,000 -> 8,000` chains and
rejects chains that fail its topology, correspondence, finite-value, or
core-frame checks. Other compatible source meshes can be used by changing the
object map in `scripts/generate_all_context_datasets.sh`.
