from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# 1. Create the model
llm = ChatOpenAI(model="gpt-4o-mini")

# 2. Define your prompt
prompt = ChatPromptTemplate.from_template("""
Use ONLY the provided context to answer the question.

Context:
{context}

Question:
{question}
""")

chain = prompt | llm

# 3. Load your context (later this will come from the retrieval agent)
#context = open("context.txt").read()
context = "walker is from california and he is a software engineer. He likes to play basketball and watch movies. He has a dog named Max. and he is currently working on a project that involves machine learning and natural language processing. He is also interested in the field of artificial intelligence and is constantly learning new things about it."
#question = input("Question: ")
question = "what is walker interested in?"

# 4. First answer
answer = chain.invoke({
    "context": context,
    "question": question
})

# 5. LOOP STARTS HERE
for i in range(3):

    critique = llm.invoke(f"""
You are reviewing an answer.

Question:
{question}

Context:
{context}

Answer:
{answer.content}

If the answer is complete, reply ONLY with:

COMPLETE

Otherwise explain what is missing.
""")

    print(f"\nIteration {i+1}")
    print("Critique:", critique.content)

    if critique.content.strip() == "COMPLETE":
        break

    answer = llm.invoke(f"""
Improve this answer using the critique below.

Context:
{context}

Question:
{question}

Current answer:
{answer.content}

Critique:
{critique.content}
""")

# 6. Final answer
print("\nFinal Answer:")
print(answer.content)
