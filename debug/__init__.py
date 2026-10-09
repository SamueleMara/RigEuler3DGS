"""Read-only visual and numerical debugging for the geometric joint pipeline.

CODEX_ADD_VISUAL_AND_NUMERICAL_DEBUG_GEOMETRIC_PIPELINE.md.

    estimator state  ->  visual + numerical debug artifacts

Section 0 rule 10: this layer must never modify an inference decision. Nothing
in this package writes back into the estimator, and `GeometricDebug` only ever
stores references to tensors the pipeline has already produced. When debugging
is disabled every entry point is a no-op, so the inference path is byte-for-byte
the same with and without it -- which section 49.9 requires as a test.
"""

from .recorder import GeometricDebug, PART_COLORS, part_color

__all__ = ["GeometricDebug", "PART_COLORS", "part_color"]
