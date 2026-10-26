# Execution Backup and Recovery

Backups preserve observations; they do not select observations or authorize a
resume. Create a backup before the first model call and at quiescent checkpoints
between studies, after a stopped run, and after finalization. This command is not
a live incremental backup service. Do not start or finalize a run concurrently.

Choose a protected destination outside the execution workspace. Its parent must
already exist. Use a separate physical device or an authorized encrypted backup
service for device-loss protection. `different_filesystem` reports filesystem
identity only: two volumes may share the same physical disk. No cloud upload or
encryption is performed by these commands.

```bash
PLAN=artifact_outputs/provider_state/canonical_campaign/provider_execution_plan.json
bash provider_run.sh backup create --plan "$PLAN" --destination /PATH/TO/NEW_BACKUP
```

The command prints the manifest path and its SHA-256. Retain that digest in a
separate trusted record. Verify and rehearse restoration without model calls:

```bash
bash provider_run.sh backup verify --backup /PATH/TO/BACKUP --manifest-sha256 DIGEST
bash provider_run.sh backup restore --backup /PATH/TO/BACKUP --manifest-sha256 DIGEST --destination /PATH/TO/NEW_WORKSPACE
```

Verification checks the inventory, every byte count and hash, and the plan and
protocol identities. Restoration refuses any existing destination, including an
empty directory. It restores file modes but never starts a worker, clears a lease,
changes the plan, or calls a model. Reinstall the pinned dependencies in the new
workspace and run the documented offline validation before considering recovery.
A stale lease or an exhausted transport allowance still requires investigation;
restoring files does not make a task eligible for another attempt.

## Selection and Integrity

The backup selects the published file inventory, the plan, and files under
`artifact_outputs/`. The only omitted output is the corpus view: its links are
validated against the fixed corpus manifest and rebuilt on restoration from the
preserved incident files. No incident content is omitted. Keep execution logs within that tree if they need to be
included. Preserve the original downloaded ZIP separately as delivery evidence.
Unlisted root files, including `.env`, caches, and virtual environments, are not
selected. A credential-named path, other symbolic link, or non-regular selected file
causes an error. A limited content check also rejects recognizable provider keys
and private-key headers; it is not a general secret detector. Operators remain
responsible for keeping credentials out of requests and logs.

The command checks fixed-input hashes, takes shared locks on existing worker
directories, and refuses an active worker without interrupting it. It compares
the complete selected inventory and hashes before and after copying. Files and
directories are synced before the completed manifest is written. A failed copy
leaves a restricted, incomplete directory without a completed manifest; it is
not a verified backup and is never reused automatically.

Backups are private operational records, not public artifact releases. They
preserve refusals, invalid outputs, timestamps, and any retained stale lease
without sanitizing or rewriting evidence. Use the separate publication command
and release checks to produce a reviewer package.
