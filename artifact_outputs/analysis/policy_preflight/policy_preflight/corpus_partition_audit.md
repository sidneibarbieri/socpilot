# Corpus Partition Audit

Aggregate audit of the canonical policy-development partitions. It exposes no incident content or source filenames.

| Partition | Incidents | Primary categories | Forensics-only profiles | Empty profiles | Corpus hash |
| --- | ---: | ---: | ---: | ---: | --- |
| Derivation | 200 | 123 | 0 | 200 | `e37fce33cbd9` |
| Held-out | 200 | 36 | 0 | 200 | `0b3f0aafd34f` |

- Total canonical incidents: `400`
- Canonical identifiers are disjoint across partitions: `true`
- Raw-to-canonical lineage: `400` raw exports, `400` canonical manifests, one-to-one.
- Primary category means the one command-execution-attempt event recorded per canonical incident.
