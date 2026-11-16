# Data and distribution

The MIT license covers this repository's original code. It does not replace
licenses or attribution requirements for downloaded data, pretrained backbones,
or third-party assets. The source repository intentionally excludes datasets
and weights. Obtain data separately from its owner and retain its license and
attributions. Generated demo data is for pipeline checks, not quality claims.

Record the source, version, preprocessing, source group, seed and partition.
Split by original image/recording before augmentation. For related recordings,
provide a common source_id (or group by speaker/session/device as appropriate).
A separate held-out dataset is required for final performance reporting.

The existing local ESC-50 download includes an Attribution-NonCommercial 3.0
license; ESC-10 has different terms. These files are not part of this source
release. Consult the upstream LICENSE before redistributing any audio.
Manifests support `path,source_id,category,duration_s`. Paths should be absolute
or relative to the project root. If source_id is omitted, files are partitioned
by path, with byte-identical copies grouped together. Independently named clips
from one recording need an explicit shared source_id.

Labeled JSONL uses `file`, optional `source_id`, `labels` for whole-recording
labels, and `segments` with `class,start_s,end_s`. Annotations must be exhaustive:
unmarked regions are treated as clean. Overlapping segments produce multi-label
frames. Recordings are partitioned before they are divided into training clips.

New ESC-50 downloads preserve the original source ID in source_id. Legacy
converted manifests without this column should be regenerated or annotated
before treating recording-level independence as established. A zero path
overlap check alone does not establish speaker/session independence.
