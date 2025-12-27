"""Research agent prompt template."""

RESEARCH_SYSTEM_PROMPT = """You are a clinical diabetes data analyst. You provide factual, concise analysis of blood glucose and insulin pump data.

## Your Role
- Analyze blood glucose and insulin data objectively
- Provide direct, factual answers
- Identify clinically relevant patterns

## Response Guidelines

### Style
- Be concise and clinical — avoid emotional language
- Answer the question directly first
- Use markdown for clarity (tables, lists, bold for key values)
- Include specific numbers; round glucose to integers, insulin to 1 decimal
- Keep responses under 150 words unless data complexity requires more

### Medical Boundaries
- You are an AI, not a medical professional
- Never recommend dosage changes
- Focus on data presentation and pattern identification

### Clinic Discussion Points
Only when you identify clinically significant patterns or concerns, add a separate section at the end:

**📋 Points for Your Clinic Visit:**
- [Specific observation with supporting data]

Add this section ONLY when you detect:
- Time in Range < 50% or Time Below Range > 10%
- Recurring patterns (e.g., consistent highs/lows at specific times)
- Significant glucose variability (CV > 40%)
- Unusual insulin patterns
- Any data suggesting potential safety concerns

Do NOT add this section for routine queries with normal findings.

### Reference Values
- **Target Range**: 70-180 mg/dL
- **Low**: < 70 mg/dL | **High**: > 180 mg/dL
- **TIR Goal**: > 70% | **Time Below Goal**: < 4%
- **CV Goal**: < 36%

## Current Context
{context}"""


def format_research_prompt(
    question: str,
    query_results: list | None = None,
    chat_history: list | None = None,
    last_error: str | None = None,
    full_data: dict | None = None,
) -> str:
    """
    Format the research agent prompt with context.

    Args:
        question: User's question
        query_results: Results from MongoDB query (if any)
        chat_history: Recent chat messages
        last_error: Any error from query execution
        full_data: Full 90-day dataset (sensorData, basalData, bolusData CSV strings)

    Returns:
        Formatted prompt string
    """
    import json

    context_parts = []

    # Add full dataset if available (for research_full and research_full_query)
    if full_data:
        context_parts.append("### Full Dataset (90 days)")

        if full_data.get("sensorData"):
            # Truncate if too long to avoid token limits
            sensor_csv = full_data["sensorData"]
            sensor_lines = sensor_csv.split("\n")
            if len(sensor_lines) > 100:
                # Keep header + first 50 + last 50 rows
                truncated = sensor_lines[:51] + ["..."] + sensor_lines[-50:]
                sensor_csv = "\n".join(truncated)
            context_parts.append("#### Sensor Glucose Data (3-hour averages)")
            context_parts.append("```csv")
            context_parts.append(sensor_csv)
            context_parts.append("```")

        if full_data.get("basalData"):
            basal_csv = full_data["basalData"]
            basal_lines = basal_csv.split("\n")
            if len(basal_lines) > 100:
                truncated = basal_lines[:51] + ["..."] + basal_lines[-50:]
                basal_csv = "\n".join(truncated)
            context_parts.append("#### Basal Rate Changes")
            context_parts.append("```csv")
            context_parts.append(basal_csv)
            context_parts.append("```")

        if full_data.get("bolusData"):
            bolus_csv = full_data["bolusData"]
            bolus_lines = bolus_csv.split("\n")
            if len(bolus_lines) > 100:
                truncated = bolus_lines[:51] + ["..."] + bolus_lines[-50:]
                bolus_csv = "\n".join(truncated)
            context_parts.append("#### Bolus and Carb Data")
            context_parts.append("```csv")
            context_parts.append(bolus_csv)
            context_parts.append("```")

    # Add query results
    if query_results:
        context_parts.append("### Query Results")
        context_parts.append("```json")
        context_parts.append(json.dumps(query_results, indent=2, default=str))
        context_parts.append("```")

    # Add chat history
    if chat_history:
        context_parts.append("\n### Recent Conversation")
        for msg in chat_history[-5:]:  # Last 5 messages
            role = "User" if msg.get("role") == "user" else "Assistant"
            text = msg.get("text", "")[:250]  # Truncate long messages
            context_parts.append(f"**{role}**: {text}")

    # Add error context
    if last_error:
        context_parts.append(f"\n### Note\nThe database query encountered an issue: {last_error}")
        context_parts.append("Please provide a helpful response based on available context.")

    # If no context available
    if not context_parts:
        context_parts.append("No additional data available. Respond based on general diabetes knowledge and the conversation history.")

    context = "\n".join(context_parts)

    return f"""User Question: {question}

{context}

Please provide a helpful, informative response."""

