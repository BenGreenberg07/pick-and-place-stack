"""Thin alias so `perception` can fit a homography without importing the
`transforms` package's public surface, keeping the import graph acyclic."""

from .transforms.homography import fit_homography as fit  # noqa: F401
