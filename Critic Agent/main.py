import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import time

# Critic Agent LLM
# (temporary: Ollama)
# Later replace with ChatOpenAI

llm = ChatOpenAI(
    model="sonar",
    api_key="",
    base_url="https://api.perplexity.ai",
    temperature=0
)

# Input from user (right now harcoded)
user_question = """
Why is my REAL ESSI simulation failing?
"""

# Temporary Agent #2 answer
agent2_answer = """
"""

critic_prompt = ChatPromptTemplate.from_template(
"""
You are a critic agent for an engineering AI system.
Evaluate the answer using your engineering knowledge.
User Question:
{question}

Agent #2 Answer:
{answer}

Score the answer from 0–100.

Approval Rule:
- If the accuracy score is 90% or higher, return Status: APPROVED.
- If the accuracy score is below 90%, return Status: FLAGGED.

Return ONLY this format:

Accuracy Score: __%

Status: APPROVED or FLAGGED


Overall:
<2 short sentence>

Do not write anything else.
"""
)



# Create final prompt
final_prompt = critic_prompt.format(
    question=user_question,
    answer=agent2_answer
)



#measure run time
start_time = time.perf_counter()
response = llm.invoke(final_prompt)
end_time = time.perf_counter()

print(response.content)

# Runtime
print("\n----- Performance -----")
print(f"Runtime: {end_time - start_time:.3f} seconds")


# Token count
try:
    usage = response.response_metadata["token_usage"]
    input_tokens = usage["prompt_tokens"]
    output_tokens = usage["completion_tokens"]
    total_tokens = usage["total_tokens"]

except Exception:
    input_tokens = None
    output_tokens = None
    total_tokens = None


# Cost Calculation

if total_tokens is not None:
    input_cost = (input_tokens / 1_000_000) * 0.25
    output_cost = (output_tokens / 1_000_000) * 2.50
    total_cost = input_cost + output_cost

else:
    input_cost = None
    output_cost = None
    total_cost = None



# CO2 Estimate
if total_tokens is not None:
    ENERGY_PER_TOKEN = 0.000000086  # kWh/token
    energy_kwh = total_tokens * ENERGY_PER_TOKEN
    CO2_PER_KWH = 0.445
    estimated_co2 = energy_kwh * CO2_PER_KWH

else:
    energy_kwh = None
    estimated_co2 = None


# Print
print(response.content)
print("\n------ Statistics ------")
print(f"Runtime: {end_time-start_time:.2f} seconds")
print(f"Input Tokens: {input_tokens}")
print(f"Output Tokens: {output_tokens}")
print(f"Total Tokens: {total_tokens}")
print(f"Input Cost: {input_cost}")
print(f"Output Cost: {output_cost}")
print(f"Total Cost: {total_cost}")
print(f"Estimated CO2: {estimated_co2}")
