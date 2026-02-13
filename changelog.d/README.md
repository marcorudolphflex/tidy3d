# Changelog Fragments

Use Towncrier fragments for user-visible changes instead of editing `CHANGELOG.md` directly.
Fragments are strongly recommended for user-facing PRs, but not enforced as a hard CI requirement.

## File naming

Create one file per change using:

`<PR_NUMBER>.<type>.md`

Examples:

- `1234.added.md`
- `1235.breaking.md`
- `1236.changed.md`
- `1237.fixed.md`

## Allowed fragment types

- `added`
- `breaking`
- `changed`
- `fixed`

## Content format

- Write plain text only (no leading `-` bullet).
- Keep it to one short sentence per fragment file.
- Focus on the user-visible impact.

Examples:

- `added`: `Added GeometryArray for efficiently representing repeated geometry instances.`
- `breaking`: `ModeSortSpec.sort_key is now required; update any code relying on None defaults.`
- `changed`: `Improved local cache performance for repeated result loads.`
- `fixed`: `Fixed race conditions when reading the local configuration directory in parallel jobs.`
