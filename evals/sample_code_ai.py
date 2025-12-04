from opik.api import get_current_trace

def run_with_metrics(user_question: str):
    # 1) Run the graph (single full run, not chunked) so it's easy to attach metrics once
    result = graph.invoke(
        {"messages": [{"role": "user", "content": user_question}]},
        config=opik_config,
    )

    final_msg = result["messages"][-1]
    output_text = final_msg.content

    # 2) Compute LLM-as-judge metrics
    jury = LLMJuriesJudge(
        judges=[
            Hallucination(model="gpt-4o-mini"),
            ComplianceRiskJudge(),
            DialogueHelpfulnessJudge(),
        ]
    )
    scores = jury.score(
        input=user_question,
        output=output_text,
    )

    # 3) Attach scores to the Opik trace as feedback / metrics
    trace = get_current_trace()  # Opik keeps track of the current trace context
    if trace is not None:
        # Option A: add one feedback object per metric
        for metric_name, metric_result in scores.items():
            trace.log_feedback(
                name=metric_name,
                value=metric_result.score,      # numeric score 0–1 or similar
                comment=metric_result.reason,   # optional explanation from the judge
            )

    return result, scores
