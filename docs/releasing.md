# Releasing and archiving

Use a tagged release to identify the exact code accompanying each manuscript or
preprint version. Do not move or rewrite a published tag.

## Release checklist

1. Confirm the working tree contains only intended changes.
2. Run the quick start, full tests, strict documentation build, and wheel build.
3. Install the wheel outside the source tree and import every package subsystem.
4. Freeze the paper-suite manifest and verify its checksum and dry-run commands.
5. Update the version, changelog or release notes, citation metadata, and known
   limitations.
6. Confirm that the Apache-2.0 license, author metadata, and copyright notice
   are included in the release artifacts.
7. Create an annotated version tag and a GitHub release.
8. Archive that release with a preservation service such as Zenodo and record
   the version DOI in `CITATION.cff` and the manuscript.
9. Build and publish the documentation for the same tag.

## Suggested version flow

- Use `0.9.x` releases while public interfaces are still changing.
- Create a dedicated tag for the preprint, for example `v0.9.0-preprint.1`.
- If review changes the model or experiments, publish a new tag rather than
  altering the preprint snapshot.
- Reserve `1.0.0` for the first explicitly stable public interface.

The author list is maintained in `pyproject.toml`. Complete the remaining
citation metadata once the preferred paper citation has been agreed by the
authors.
