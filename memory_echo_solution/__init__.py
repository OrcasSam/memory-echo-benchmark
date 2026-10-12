"""Strict cumulative validation of the Memory Echo governance lifecycle."""

ARMS = (
    "baseline",
    "s1_identity",
    "s2_retrieval_hygiene",
    "s3_reconsolidation",
    "s4_persistence",
    "full_guard",
)

ARM_LEVEL = {arm: index for index, arm in enumerate(ARMS)}


def has_level(arm: str, minimum: str) -> bool:
    return ARM_LEVEL[arm] >= ARM_LEVEL[minimum]


IDENTITY_ARMS = {arm for arm in ARMS if has_level(arm, "s1_identity")}
RETRIEVAL_HYGIENE_ARMS = {arm for arm in ARMS if has_level(arm, "s2_retrieval_hygiene")}
RECONSOLIDATION_ARMS = {arm for arm in ARMS if has_level(arm, "s3_reconsolidation")}
PERSISTENCE_ARMS = {arm for arm in ARMS if has_level(arm, "s4_persistence")}
COUNTERFACTUAL_DECISION_ARMS = {"full_guard"}
