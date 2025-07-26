from langchain.prompts import PromptTemplate


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