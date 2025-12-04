QUERIES = {
  "REWARD_HACKING": "What does Lilian Weng say about types of reward hacking?",
  "HALLUCINATION": "What does Lilian Weng say about causes of hallucination?",
}


# Golden answers provide the ground-truth snippets used when logging Opik metrics.
EXPECTED_ANSWERS = {
  "REWARD_HACKING": (
    "Lilian Weng splits reward hacking into two broad classes: misspecified goals or "
    "environments where the proxy reward fails to reflect the task, and reward tampering "
    "(a.k.a. wireheading) where the agent manipulates or shortcuts the reward channel "
    "instead of solving the task."
  ),
  "HALLUCINATION": (
    "She attributes hallucinations to knowledge gaps and brittle reasoning (insufficient or "
    "outdated training data), to decoding dynamics such as high temperature sampling, and to "
    "misaligned incentives where the model prioritizes fluent answers over factual grounding."
  ),
}
