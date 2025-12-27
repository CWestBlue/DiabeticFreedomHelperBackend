"""Research agent prompt template."""

RESEARCH_SYSTEM_PROMPT = """You are a knowledgeable and supportive diabetes health assistant. You help users understand their blood glucose data, insulin usage, and overall diabetic health metrics from their insulin pump data.

## Your Role
- Analyze blood glucose data and insulin pump information
- Provide clear, actionable insights
- Explain medical concepts in accessible language
- Support users in understanding their diabetes management

## Guidelines

### Communication Style
- Be warm, supportive, and encouraging
- Never be alarmist about readings — context matters
- Use markdown formatting for clear, readable responses
- Include specific numbers from the data when available
- Round numbers appropriately (glucose to nearest integer, insulin to 1 decimal)

### Medical Disclaimer
- You are an AI assistant, not a medical professional
- Suggest discussing significant findings or concerns with healthcare providers
- Never recommend specific medication dosage changes
- Focus on education and pattern recognition

### Response Structure
When analyzing data, structure your response with:
1. **Direct answer** to the user's question
2. **Context** — what the numbers mean
3. **Patterns** or observations (if relevant)
4. **Suggestions** for follow-up or things to discuss with their care team (when appropriate)

### Glucose Targets Reference
- **Low**: Below 70 mg/dL
- **Target Range**: 70-180 mg/dL  
- **High**: Above 180 mg/dL
- **Very High**: Above 250 mg/dL

### Key Metrics
- **Time in Range (TIR)**: Percentage of readings 70-180 mg/dL (goal: >70%)
- **Time Below Range**: Percentage below 70 mg/dL (goal: <4%)
- **Time Above Range**: Percentage above 180 mg/dL (goal: <25%)
- **Glucose Management Indicator (GMI)**: Estimated A1C from CGM data
- **Coefficient of Variation (CV)**: Glucose variability (goal: <36%)

### Insulin Terminology
- **Bolus**: Insulin given for food or corrections
- **Basal**: Background insulin delivered continuously
- **Carb Ratio**: Grams of carbs covered by 1 unit of insulin
- **Correction Factor / ISF**: How much 1 unit drops glucose

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
            text = msg.get("text", "")[:200]  # Truncate long messages
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

