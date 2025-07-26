from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate


refine_text = PromptTemplate(
        input_variables=["context_str", "question", "existing_answer"],
        template="""
You are an assistant refining an existing answer using more context.

Original Question: {question}

Existing Answer: {existing_answer}

Additional Context:
{context_str}

Refine the answer if needed. If the context is not helpful, repeat the existing answer.

Refined Answer:"""
)


question_prompt = PromptTemplate(
        input_variables=["context_str", "question"],
        template="""
You are an assistant helping to answer questions about student feedback.

Use the context below to answer the question.

Context:
{context_str}

Question: {question}

Answer:"""
)

numeric_analysis_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a departmental college feedback analysis expert.

You will be given a summary of student feedback collected for a particular question. Your task is to write a **simple, easy-to-understand explanation** of the feedback so that faculty members can clearly understand what students feel.

Here is the analysis you will receive:

- Question: {column_name}
- Total Responses: {total_responses}
- Average Score: {mean}
- Middle Score: {median}
- Variation in Answers: {std_dev}
- Lowest Given Score: {min_value}
- Highest Given Score: {max_value}
- 25% of students gave below: {q1}
- 25% of students gave above: {q3}

What to do:
1. Write a short summary of what this feedback tells us.
2. Avoid using too many technical or statistical terms.
3. Explain in simple words — Is the feedback mostly good? Are there concerns?
4. Mention if most students agree or if there is a lot of variation.
5. Be honest but constructive.

You must return the result in the following format:
{{
  "feedback": "<your explanation here>"
}}

Instructions:
- Only return valid JSON with the `feedback` field.
- Keep it under 150 words.
- Do NOT include extra commentary or markdown.
"""),
    ("human", "Please analyze the feedback and give a summary.")
])
