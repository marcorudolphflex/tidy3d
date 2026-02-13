#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

TYPE_TO_SECTION = {
    "added": "Added",
    "changed": "Changed",
    "fixed": "Fixed",
    "removed": "Removed",
    "breaking": "Breaking Changes",
    "planned_deprecation": "Planned Deprecation",
}


@dataclass
class UnreleasedEntry:
    line: int
    section: str
    text: str
    normalized_text: str


@dataclass
class FragmentMapping:
    source: str
    fragment_type: str | None
    section: str | None
    text: str
    normalized_text: str
    changelog_line: int | None
    blame_commit: str | None
    pr_number: int | None
    target: str | None
    status: str
    detail: str | None = None


def _run_command(*args: str, check: bool = True) -> str:
    completed = subprocess.run(
        args,
        check=check,
        text=True,
        capture_output=True,
    )
    return completed.stdout.strip()


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def _load_changelog_at_ref(ref: str, changelog_path: Path) -> list[str]:
    content = _run_command("git", "show", f"{ref}:{changelog_path.as_posix()}")
    return content.splitlines()


def _parse_unreleased_entries(lines: list[str]) -> list[UnreleasedEntry]:
    unreleased_start = None
    for idx, line in enumerate(lines, start=1):
        if line.strip() == "## [Unreleased]":
            unreleased_start = idx
            break

    if unreleased_start is None:
        raise RuntimeError("Could not find '## [Unreleased]' in referenced changelog.")

    entries: list[UnreleasedEntry] = []
    current_section: str | None = None

    for idx in range(unreleased_start + 1, len(lines) + 1):
        line = lines[idx - 1]
        if idx > unreleased_start + 1 and line.startswith("## ["):
            break

        section_match = re.match(r"^###\s+(.+?)\s*$", line)
        if section_match:
            current_section = section_match.group(1)
            continue

        if line.startswith("- ") and current_section:
            text = line[2:].strip()
            entries.append(
                UnreleasedEntry(
                    line=idx,
                    section=current_section,
                    text=text,
                    normalized_text=_normalize_text(text),
                )
            )

    return entries


def _parse_fragment_type(filename: str) -> str | None:
    match = re.match(r"^.+\.([a-z_]+)\.md$", filename)
    if match:
        return match.group(1)
    return None


def _resolve_pr_for_commit(commit: str, repo: str, cache: dict[str, int | None]) -> int | None:
    if commit in cache:
        return cache[commit]

    try:
        output = _run_command(
            "gh",
            "api",
            f"repos/{repo}/commits/{commit}/pulls",
            "--jq",
            ".[0].number",
            check=True,
        )
    except subprocess.CalledProcessError:
        cache[commit] = None
        return None

    if not output:
        cache[commit] = None
        return None

    try:
        number = int(output)
    except ValueError:
        number = None

    cache[commit] = number
    return number


def _blame_commit_for_line(ref: str, changelog_path: Path, line_number: int) -> str:
    output = _run_command(
        "git",
        "blame",
        "-L",
        f"{line_number},{line_number}",
        ref,
        "--",
        changelog_path.as_posix(),
    )
    commit = output.split()[0].lstrip("^")
    if not re.fullmatch(r"[0-9a-fA-F]{7,40}", commit):
        raise RuntimeError(f"Could not parse blame commit from line: {output}")
    return commit


def _map_fragments(
    fragments_dir: Path,
    fragment_glob: str,
    unreleased_entries: list[UnreleasedEntry],
    changelog_ref: str,
    changelog_path: Path,
    repo: str,
) -> list[FragmentMapping]:
    by_section: dict[str, list[UnreleasedEntry]] = {}
    for entry in unreleased_entries:
        by_section.setdefault(entry.section, []).append(entry)

    used_line_numbers: set[int] = set()
    pr_cache: dict[str, int | None] = {}
    mappings: list[FragmentMapping] = []

    for fragment_path in sorted(fragments_dir.glob(fragment_glob)):
        if fragment_path.name in {"README.md", "template.md"}:
            continue
        if not fragment_path.is_file():
            continue

        text = fragment_path.read_text(encoding="utf-8").strip()
        normalized_text = _normalize_text(text)
        fragment_type = _parse_fragment_type(fragment_path.name)
        section = TYPE_TO_SECTION.get(fragment_type or "")

        mapping = FragmentMapping(
            source=fragment_path.name,
            fragment_type=fragment_type,
            section=section,
            text=text,
            normalized_text=normalized_text,
            changelog_line=None,
            blame_commit=None,
            pr_number=None,
            target=None,
            status="unmapped",
        )

        if not fragment_type:
            mapping.status = "invalid_filename"
            mapping.detail = "Could not parse fragment type from filename."
            mappings.append(mapping)
            continue

        if section is None:
            mapping.status = "unsupported_type"
            mapping.detail = f"No changelog section mapping for type '{fragment_type}'."
            mappings.append(mapping)
            continue

        section_entries = by_section.get(section, [])
        candidates = [
            entry for entry in section_entries if entry.normalized_text == normalized_text
        ]
        if not candidates:
            mapping.status = "no_text_match"
            mapping.detail = "No matching bullet found in Unreleased section."
            mappings.append(mapping)
            continue

        chosen = next(
            (entry for entry in candidates if entry.line not in used_line_numbers), candidates[0]
        )
        used_line_numbers.add(chosen.line)

        mapping.changelog_line = chosen.line
        try:
            commit = _blame_commit_for_line(changelog_ref, changelog_path, chosen.line)
        except RuntimeError as error:
            mapping.status = "blame_error"
            mapping.detail = str(error)
            mappings.append(mapping)
            continue

        mapping.blame_commit = commit
        pr_number = _resolve_pr_for_commit(commit=commit, repo=repo, cache=pr_cache)
        if pr_number is None:
            mapping.status = "no_pr"
            mapping.detail = f"No PR associated with commit {commit}."
            mappings.append(mapping)
            continue

        mapping.pr_number = pr_number
        mapping.target = f"{pr_number}.{fragment_type}.md"
        mapping.status = "mapped"
        mappings.append(mapping)

    return mappings


def _apply_renames(
    fragments_dir: Path,
    mappings: list[FragmentMapping],
    apply: bool,
    collision_mode: str,
) -> list[FragmentMapping]:
    target_to_mappings: dict[str, list[FragmentMapping]] = {}
    source_names = {mapping.source for mapping in mappings}

    for mapping in mappings:
        if mapping.status == "mapped" and mapping.target:
            target_to_mappings.setdefault(mapping.target, []).append(mapping)

    for target, group in target_to_mappings.items():
        if len(group) > 1:
            if collision_mode == "numbered":
                target_match = re.fullmatch(r"(\d+)\.([a-z_]+)\.md", target)
                if not target_match:
                    for mapping in group:
                        mapping.status = "collision"
                        mapping.detail = (
                            f"Target '{target}' is shared by {len(group)} fragments "
                            "and cannot be converted to numbered targets."
                        )
                    continue

                pr_number, fragment_type = target_match.groups()
                sorted_group = sorted(
                    group,
                    key=lambda mapping: (
                        mapping.changelog_line or 0,
                        mapping.source,
                    ),
                )
                for index, mapping in enumerate(sorted_group, start=1):
                    numbered_target = f"{pr_number}.{index}.{fragment_type}.md"
                    mapping.target = numbered_target
                    source_path = fragments_dir / mapping.source
                    target_path = fragments_dir / numbered_target

                    if source_path.name == target_path.name:
                        mapping.status = "already_named"
                        mapping.detail = "Source already uses target name."
                        continue

                    if target_path.exists() and target_path.name not in source_names:
                        mapping.status = "target_exists"
                        mapping.detail = f"Target '{numbered_target}' already exists."
                        continue

                    if apply:
                        source_path.rename(target_path)
                        mapping.status = "renamed"
                        mapping.detail = f"Renamed to '{numbered_target}'."
                    else:
                        mapping.status = "would_rename"
                        mapping.detail = f"Would rename to '{numbered_target}'."
            else:
                for mapping in group:
                    mapping.status = "collision"
                    mapping.detail = f"Target '{target}' is shared by {len(group)} fragments."
            continue

        mapping = group[0]
        source_path = fragments_dir / mapping.source
        target_path = fragments_dir / target

        if source_path.name == target_path.name:
            mapping.status = "already_named"
            mapping.detail = "Source already uses target name."
            continue

        if target_path.exists() and target_path.name not in source_names:
            mapping.status = "target_exists"
            mapping.detail = f"Target '{target}' already exists."
            continue

        if apply:
            source_path.rename(target_path)
            mapping.status = "renamed"
            mapping.detail = f"Renamed to '{target}'."
        else:
            mapping.status = "would_rename"
            mapping.detail = f"Would rename to '{target}'."

    return mappings


def _print_summary(mappings: list[FragmentMapping]) -> None:
    counts: dict[str, int] = {}
    for mapping in mappings:
        counts[mapping.status] = counts.get(mapping.status, 0) + 1

    print("Summary:")
    for status in sorted(counts):
        print(f"  {status}: {counts[status]}")

    print("\nMappings:")
    for mapping in mappings:
        print(
            "\t".join(
                [
                    mapping.source,
                    mapping.fragment_type or "",
                    mapping.section or "",
                    str(mapping.changelog_line or ""),
                    mapping.blame_commit or "",
                    str(mapping.pr_number or ""),
                    mapping.target or "",
                    mapping.status,
                    mapping.detail or "",
                ]
            )
        )


def _write_report(mappings: list[FragmentMapping], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_data: list[dict[str, Any]] = [asdict(mapping) for mapping in mappings]
    report_path.write_text(json.dumps(report_data, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Map migration changelog fragments to PR numbers using "
            "git blame on develop and GitHub commit->PR lookup."
        )
    )
    parser.add_argument(
        "--ref", default="upstream/develop", help="Git ref used for changelog blame."
    )
    parser.add_argument(
        "--repo", default="flexcompute/tidy3d", help="GitHub repository owner/name."
    )
    parser.add_argument(
        "--changelog-path",
        default="CHANGELOG.md",
        help="Path to changelog file inside the repository.",
    )
    parser.add_argument(
        "--fragments-dir",
        default="changelog.d",
        help="Directory containing Towncrier fragments.",
    )
    parser.add_argument(
        "--fragment-glob",
        default="migration-*.md",
        help="Glob used to select fragments to map.",
    )
    parser.add_argument(
        "--report",
        default="playground/changelog_fragment_pr_mapper_report.json",
        help="Path to write JSON report.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply renames for unambiguous mapped fragments.",
    )
    parser.add_argument(
        "--collision-mode",
        choices=("leave", "numbered"),
        default="leave",
        help=(
            "How to handle multiple fragments mapping to the same '<PR>.<type>.md' "
            "target. 'numbered' rewrites them as '<PR>.<N>.<type>.md'."
        ),
    )
    args = parser.parse_args()

    changelog_path = Path(args.changelog_path)
    fragments_dir = Path(args.fragments_dir)
    report_path = Path(args.report)

    if not changelog_path.exists():
        raise FileNotFoundError(f"Changelog does not exist: {changelog_path}")
    if not fragments_dir.exists():
        raise FileNotFoundError(f"Fragments dir does not exist: {fragments_dir}")

    unreleased_lines = _load_changelog_at_ref(ref=args.ref, changelog_path=changelog_path)
    unreleased_entries = _parse_unreleased_entries(unreleased_lines)
    mappings = _map_fragments(
        fragments_dir=fragments_dir,
        fragment_glob=args.fragment_glob,
        unreleased_entries=unreleased_entries,
        changelog_ref=args.ref,
        changelog_path=changelog_path,
        repo=args.repo,
    )
    mappings = _apply_renames(
        fragments_dir=fragments_dir,
        mappings=mappings,
        apply=args.apply,
        collision_mode=args.collision_mode,
    )
    _print_summary(mappings)
    _write_report(mappings=mappings, report_path=report_path)

    if any(
        mapping.status in {"invalid_filename", "unsupported_type", "blame_error"}
        for mapping in mappings
    ):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
