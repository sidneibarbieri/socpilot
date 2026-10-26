from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from soc_llm_policy.paths import RepoPaths, resolve_repo_root


def _copy_file(src: Path, dst: Path, *, overwrite: bool) -> None:
    if dst.exists() and not overwrite:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_tree(src: Path, dst: Path, *, overwrite: bool) -> None:
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            _copy_tree(item, target, overwrite=overwrite)
        else:
            _copy_file(item, target, overwrite=overwrite)


def import_package(
    *,
    repo_paths: RepoPaths,
    package_root: Path,
    overwrite: bool,
) -> None:
    dataset_root = package_root / "dataset"
    global_root = package_root / "global"

    if not dataset_root.exists():
        raise FileNotFoundError(f"Missing directory: {dataset_root}")
    if not global_root.exists():
        raise FileNotFoundError(f"Missing directory: {global_root}")

    catalog_src = global_root / "action_catalog.yaml"
    constraints_src = global_root / "constraints.yaml"
    if not catalog_src.exists():
        raise FileNotFoundError(f"Missing file: {catalog_src}")
    if not constraints_src.exists():
        raise FileNotFoundError(f"Missing file: {constraints_src}")

    _copy_file(catalog_src, repo_paths.inbox_action_catalog_path, overwrite=overwrite)
    _copy_file(
        constraints_src,
        repo_paths.inbox_constraints_path,
        overwrite=overwrite,
    )

    playbook_src = global_root / "playbook"
    playbook_dst = repo_paths.inbox_global_dir / "playbook"
    _copy_tree(playbook_src, playbook_dst, overwrite=overwrite)

    incident_dirs = sorted(path for path in dataset_root.glob("INC_*") if path.is_dir())
    if not incident_dirs:
        raise FileNotFoundError(f"No incidents found in {dataset_root}/INC_*")

    for incident_dir in incident_dirs:
        dst_dir = repo_paths.inbox_incident_dir(incident_dir.name)
        _copy_tree(incident_dir, dst_dir, overwrite=overwrite)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="soc_llm_policy.dataset_import")
    parser.add_argument("--repo-root", default=None)
    parser.add_argument(
        "--package-root",
        required=True,
        help="Root directory containing dataset/ and global/ subfolders.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in inbox.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    repo_root = resolve_repo_root(args.repo_root)
    paths = RepoPaths(repo_root=repo_root)
    package_root = Path(args.package_root).expanduser().resolve()
    import_package(
        repo_paths=paths,
        package_root=package_root,
        overwrite=bool(args.overwrite),
    )
    print(f"Package imported to: {paths.inbox_dir}")


if __name__ == "__main__":
    main()
