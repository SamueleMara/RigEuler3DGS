"""Unified bidirectional Phase-1 free-motion IEKF / IEKS.

CODEX_IMPLEMENT_EXACT_UNIFIED_PHASE1_IEKF_IEKS.md.

Opt-in. Nothing in this package is imported by the existing forward pipeline,
and nothing here rewrites the working free or rigid estimators: the process
model, renderer bridge, measurement precision and SO(3) manifold come from
`estimator/` unchanged, and this package adds the pieces that did not exist --
two-source candidate indexing, the exact nonlinear process-residual
Jacobians, and the single full-window posterior that couples frames.

WHY FORWARD AND BACKWARD INPUTS HAVE THESE SPECIFIC ROLES (section 9, which
the specification requires to appear in the code and not only in a report):

1. The forward source population is trained at original t=0 and is therefore
   geometrically anchored there; the forward free IEKF initialises those
   states.
2. The backward source population is trained at original t=T and introduces
   genuinely represented final-endpoint Gaussian states -- surfaces missing
   from the initial endpoint. It is NOT a more authoritative posterior.
3. Numerical time direction is not a physical preference. The unified
   physical-time posterior does not depend on the order its factors are
   eliminated in; forward elimination then backward substitution is a way of
   solving that posterior, not an asymmetric fusion rule.
4. Image evidence is never duplicated. Both preliminary IEKFs saw (largely)
   the same RGB-D frames. The unified optimiser uses each image once and
   takes the preliminary trajectories as initialisation only.
5. Occluded early states stay uncertain. A candidate with no contributing
   pixels has a zero instantaneous image Jacobian; information reaches it
   only through the process coupling and co-visible relations. An
   unidentifiable trajectory must not be given a confident motion or label.
6. Swapping endpoint names must not change the physical problem. Solving the
   same objective from a forward-first or a backward-first nominal should
   reach the same posterior when the optimiser finds the same local optimum.
   That is tested by objective and alignment, not by bitwise equality.
"""

from .provenance import (  # noqa: F401
    INITIAL,
    FINAL,
    CandidatePopulation,
    SourceAnchor,
    candidate_manifest,
)
from .free_dynamics import (  # noqa: F401
    reverse_to_physical_state,
    reverse_to_physical_covariance,
    verify_transition_against_finite_differences,
)
from .process_factors import (  # noqa: F401
    process_residual,
    process_residual_jacobians,
)
