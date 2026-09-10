"""Modular feature definitions with a planned, verified execution.

Write each feature as its own function and you keep the modularity that makes
experimentation possible; the planner recovers the performance that writing
them separately would otherwise cost. See ``docs/usage_docs`` for the guide.
"""

from prism_eda.features.algebra import (
    Expr,
    TracingError,
    Window,
    distinct_ratio,
    guard,
    lag,
    lead,
    log1p,
    safe_div,
    time_delta,
    top_share,
)
from prism_eda.features.contract import (
    ContractViolation,
    FeatureContract,
    FeatureSpec,
)
from prism_eda.features.featureset import FeatureSet
from prism_eda.features.ir import Level, Node
from prism_eda.features.plan import FeaturePlan
from prism_eda.features.results import FeaturePlanResult, FeatureRun
from prism_eda.features.tracing import TracedFeature
from prism_eda.features.verify import (
    FeatureAgreement,
    VerificationError,
    VerificationReport,
)

__all__ = [
    "ContractViolation",
    "Expr",
    "FeatureAgreement",
    "FeatureContract",
    "FeaturePlan",
    "FeaturePlanResult",
    "FeatureRun",
    "FeatureSet",
    "FeatureSpec",
    "Level",
    "Node",
    "TracedFeature",
    "TracingError",
    "VerificationError",
    "VerificationReport",
    "Window",
    "distinct_ratio",
    "guard",
    "lag",
    "lead",
    "log1p",
    "safe_div",
    "time_delta",
    "top_share",
]
