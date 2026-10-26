# Cross-Batch Policy Activation

Generated: `2026-09-19T12:54:39.568421+00:00`

- Policy-development batch: `200` incidents (`e37fce33cbd9`)
- Transfer-check batch: `200` incidents (`0b3f0aafd34f`)
- Constraints: `policy/constraints.yaml` (`932eb36d686c`)

## Content-triggered rules (transfer evidence)

| Rule | Action | Policy development | Transfer check | Meets floor |
| --- | --- | --- | --- | --- |
| shadowcopy_tamper_isolate | isolate_host | 17 (0.09) | 54 (0.27) | yes |
| intel_match_block | block_egress | 19 (0.10) | 92 (0.46) | yes |
| sensor_tamper_isolate | isolate_host | 6 (0.03) | 3 (0.01) | yes |
| remote_access_block | block_egress | 6 (0.03) | 9 (0.04) | yes |
| malware_detect_quarantine | quarantine_artifact | 59 (0.29) | 10 (0.05) | yes |
| account_probe_block | block_egress | 9 (0.04) | 9 (0.04) | yes |
| ops_monitor_no_isolate | isolate_host | 7 (0.04) | 17 (0.09) | yes |
| ops_monitor_no_reimage | reimage_format_host | 7 (0.04) | 17 (0.09) | yes |
| creddump_reset | reset_disable_account | 9 (0.04) | 1 (0.01) | no |
| evasion_isolate | isolate_host | 17 (0.09) | 2 (0.01) | no |
| exploit_waf_rule | apply_ips_waf_rule | 17 (0.09) | 2 (0.01) | no |
| persistence_quarantine | quarantine_artifact | 8 (0.04) | 0 (0.00) | no |
| recon_block | block_egress | 13 (0.07) | 1 (0.01) | no |
| c2_block | block_egress | 4 (0.02) | 1 (0.01) | no |

A content-triggered rule meets the transfer floor when it activates on at least 3 transfer-check incidents. Rules below the floor remain local operational rules and do not support the transfer result.

## Global rules (always-applicable policy, not coverage)

| Rule | Action | Policy development | Transfer check | Meets floor |
| --- | --- | --- | --- | --- |
| evidence_before_reimage | reimage_format_host | 200 (1.00) | 200 (1.00) | no |
| approve_isolate | isolate_host | 200 (1.00) | 200 (1.00) | no |
| approve_reimage | reimage_format_host | 200 (1.00) | 200 (1.00) | no |
| approve_lift_isolation | lift_isolation | 200 (1.00) | 200 (1.00) | no |
| approve_account_disable | reset_disable_account | 200 (1.00) | 200 (1.00) | no |

Global rules match every incident by construction. Their support is reported for completeness only and is never counted as coverage evidence, which would inflate transfer support.
