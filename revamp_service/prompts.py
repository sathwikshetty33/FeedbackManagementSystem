from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate


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
    ("system", """You are a college feedback analysis expert. You will analyze student feedback data step by step.

IMPORTANT: You must return ONLY valid JSON. No extra text, no explanations outside the JSON.

Given data:
- Question: {column_heading}
- Total Responses: {total_responses}
- Average Score: {mean}
- Middle Score: {median}
- Variation in Answers: {std_dev}
- Lowest Score: {min_value}
- Highest Score: {max_value}
- 25% scored below: {q1}
- 75% scored below: {q3}

Think step by step:

Step 1: Look at the average score ({mean}). 
- If 4.0+ = Very Good
- If 3.0-3.9 = Good  
- If 2.0-2.9 = Fair
- If below 2.0 = Poor

Step 2: Check if students mostly agree by looking at variation ({std_dev}).
- If below 0.8 = Most students agree
- If 0.8-1.2 = Some disagreement
- If above 1.2 = Students have mixed opinions

Step 3: Look at the range from {min_value} to {max_value}.
- Wide range = Students have very different opinions
- Narrow range = Students mostly think similarly

Step 4: Write a simple summary using these observations.

Return format (copy exactly):
{{
  "feedback": "write your analysis here in simple words"
}}

Rules:
- Maximum 100 words
- Use simple language
- No technical terms
- Only return the JSON object
- No extra text before or after"""),
    ("human", "Analyze this feedback data step by step and return only JSON:")
])

# Alternative version with even more structure for better results:
numeric_analysis_prompt_v2 = ChatPromptTemplate.from_messages([
    ("system", """You analyze student feedback. Follow these exact steps:

DATA:
Question: {column_heading}
Responses: {total_responses}
Average: {mean}
Lowest: {min_value} | Highest: {max_value}
Variation: {std_dev}

STEPS:
1. Score Level: {mean} means feedback is [Very Good/Good/Fair/Poor]
2. Agreement: {std_dev} shows [High/Medium/Low] agreement among students
3. Summary: Write 2-3 sentences explaining what this means for faculty

OUTPUT FORMAT (copy exactly):
{{
  "feedback": "Your analysis here"
}}

RULES:
- Return ONLY the JSON object
- Maximum 500 words
- No markdown, no extra text
- Simple language only"""),
    ("human", "Please analyze:")
])

# Even simpler version with forced structure:
numeric_analysis_prompt_v3 = ChatPromptTemplate.from_messages([
    ("system", """Analyze student feedback data. Return only JSON.

Data: Question={column_heading}, Responses={total_responses}, Average={mean}, Range={min_value}-{max_value}, Variation={std_dev}

Think:
- Average {mean}: Is this good (3.5+), okay (2.5-3.4), or concerning (below 2.5)?
- Variation {std_dev}: Do students agree (under 1.0) or disagree (over 1.0)?

Write summary in 50 words maximum.

Response format:
{{
  "feedback": "Students gave an average of {mean}. This suggests [positive/mixed/negative] feedback. Students [mostly agree/have mixed opinions] based on the variation of {std_dev}. Overall, this indicates [brief conclusion]."
}}"""),
    ("human", "Analyze the data:")
])

# Most constrained version - best for small models:
numeric_analysis_prompt_final = ChatPromptTemplate.from_messages([
    ("system", """You are a feedback analyzer. Return only valid JSON.

Given: Average={mean}, Responses={total_responses}, Variation={std_dev}

Template response:
{{
  "feedback": "Students gave an average score of {mean} out of [scale]. This is [positive/concerning] feedback. The variation of {std_dev} shows students [mostly agree/have mixed opinions]. Faculty should [action/note]."
}}

Replace bracketed items with appropriate words. Keep under 60 words total.
Return only the JSON object."""),
    ("human", "Analyze:")
])

numeric_analysis_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a feedback analysis expert. Analyze the data step-by-step and be logically consistent.

CRITICAL RULES:
1. If standard deviation is HIGH (above 2.0), students DISAGREE significantly
2. If standard deviation is LOW (below 1.0), students mostly AGREE  
3. If standard deviation is MEDIUM (1.0-2.0), there is moderate disagreement
4. NEVER say students agree when standard deviation is high
5. NEVER contradict yourself in the same response

DATA PROVIDED:
- Question: {column_heading}
- Total Responses: {total_responses}
- Average Score: {mean}
- Standard Deviation: {std_dev}
- Lowest Score: {min_value}
- Highest Score: {max_value}
- 25th percentile: {q1}
- 75th percentile: {q3}

STEP-BY-STEP ANALYSIS:

Step 1: Interpret the average score
- Look at {mean} and determine if this is positive, neutral, or concerning

Step 2: Analyze student agreement 
- Look at {std_dev} number carefully
- If it's above 2.0 = students have very different opinions
- If it's 1.0-2.0 = students have somewhat different opinions  
- If it's below 1.0 = students mostly agree

Step 3: Consider the range
- Look at {min_value} to {max_value} range
- Wide range + high std dev = confirms disagreement
- Narrow range + low std dev = confirms agreement

Step 4: Write summary
- Be consistent with your analysis from steps 1-3
- Don't contradict the standard deviation interpretation
- Explain what faculty should understand

Return ONLY this JSON format:
{{
  "feedback": "Your analysis here - maximum 120 words"
}}

Remember: High standard deviation = disagreement. Low standard deviation = agreement. Be consistent."""),
    ("human", "Analyze this feedback data following the steps above:")
])

# Alternative version with even clearer logic checks
numeric_analysis_prompt_v2 = ChatPromptTemplate.from_messages([
    ("system", """You analyze student feedback. Follow this logic exactly:

Given data:
- Question: {column_heading}  
- Average: {mean}
- Standard Deviation: {std_dev}
- Range: {min_value} to {max_value}
- Responses: {total_responses}

LOGIC CHECK:
Before writing anything, ask yourself:
1. Is {std_dev} greater than 2.0? If YES → students disagree significantly
2. Is {std_dev} between 1.0-2.0? If YES → students have mixed opinions
3. Is {std_dev} less than 1.0? If YES → students mostly agree

NEVER write that students "agree" or "think similarly" if the standard deviation is above 1.5.

Write your analysis explaining:
- What the average score means
- Whether students agree or disagree (based on std dev)
- What this tells faculty

Format: {{"feedback": "your analysis"}}

Double-check: Does your response match the standard deviation value?"""),
    ("human", "Please analyze this data:")
])

# Most explicit version
numeric_analysis_prompt_v3 = ChatPromptTemplate.from_messages([
    ("system", """Analyze student feedback data. Be mathematically accurate.

Data: Average={mean}, Standard Deviation={std_dev}, Range={min_value}-{max_value}, N={total_responses}

INTERPRETATION GUIDE:
- Standard deviation {std_dev}: 
  * Above 2.0 = HIGH disagreement among students
  * 1.0 to 2.0 = MODERATE disagreement  
  * Below 1.0 = LOW disagreement (students agree)

YOUR TASK:
1. Comment on the average score {mean}
2. State whether students agree or disagree based on std dev {std_dev}
3. Mention what the range {min_value}-{max_value} tells us
4. Give faculty a clear takeaway

CRITICAL: Your description of student agreement MUST match the standard deviation value.

Return: {{"feedback": "analysis in simple language"}}"""),
    ("human", "Analyze:")
])

# Validation-focused version
numeric_analysis_prompt_final = ChatPromptTemplate.from_messages([
    ("system", """You are analyzing student feedback. Be precise and logical.

Question: {column_heading}
Average Score: {mean}
Standard Deviation: {std_dev} 
Score Range: {min_value} to {max_value}
Total Responses: {total_responses}

MANDATORY LOGIC:
- Standard deviation of {std_dev} means students [agree/disagree] significantly
- You must correctly interpret whether {std_dev} is high or low
- High std dev (>2.0) = disagreement, Low std dev (<1.0) = agreement

Write a brief analysis that:
1. Evaluates the {mean} average score
2. CORRECTLY interprets the {std_dev} standard deviation  
3. Explains what faculty should know
4. Stays under 100 words

Return only: {{"feedback": "your analysis"}}

Before responding, verify: Does my interpretation of {std_dev} make mathematical sense?"""),
    ("human", "Analyze the feedback:")
])

categorical_analysis_prompt = ChatPromptTemplate.from_messages([
    ("system", """You analyze student feedback for categorical responses. Follow this logic exactly:

Given data:
- Question: {column_heading}
- Total Responses: {total_responses}
- Number of Different Categories: {unique_categories}
- Distribution: {distribution}
- Percentages: {percentages}
- Most Common Response: {most_common}
- Least Common Response: {least_common}

LOGIC CHECK FOR CATEGORICAL DATA:
Before writing anything, analyze the distribution:

1. CONSENSUS CHECK: Look at the most common response percentage
   - If top response > 70% → Strong consensus among students
   - If top response 50-70% → Moderate consensus
   - If top response < 50% → No clear consensus (opinions are divided)

2. DISTRIBUTION CHECK: Look at how responses are spread
   - If 2-3 categories dominate (>80% combined) → Clear patterns
   - If responses are spread across many categories → Very mixed opinions
   - If similar percentages across categories → No dominant view

3. DIVERSITY CHECK: Consider number of unique categories vs responses
   - Many categories with responses → Students have varied perspectives
   - Few categories → Students think in similar ways

CRITICAL RULES:
- If the top response is less than 50%, NEVER say "most students agree"
- If responses are spread across many categories, acknowledge the diversity
- Be specific about what the most common response tells us
- Mention concerning patterns if they exist

Write your analysis explaining:
- What the most common response ({most_common}) tells us
- Whether students have consensus or divided opinions
- What the distribution pattern means for faculty
- Any notable insights from the percentages

Format: {{"feedback": "your analysis"}}

Double-check: Does your interpretation match the actual distribution percentages?"""),
    ("human", "Please analyze this categorical feedback data:")
])

chatbot_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an AI assistant specializing in feedback analysis for the Department of Artificial Intelligence and Machine Learning (AIML) at Dayananda Sagar College of Engineering.

Your primary role is to analyze student and faculty feedback data, but you should also respond naturally to general conversational inputs like greetings, thanks, and casual questions.

Description: {description}

INSTRUCTIONS:
1. For feedback analysis questions:
   - Use the provided context to answer questions about feedback data
   - If the required information is not in the context, explicitly state: "The answer cannot be determined from the provided feedback data."
   - Provide specific insights, trends, and actionable recommendations when possible
   - Reference specific data points from the context when available

2. For general conversational inputs (greetings, thanks, casual questions):
   - Respond naturally and helpfully
   - Maintain a professional but friendly tone
   - If appropriate, guide the conversation back to how you can help with feedback analysis

3. For unclear or ambiguous inputs:
   - Ask clarifying questions to better understand what the user needs
   - Offer to help with feedback analysis if the intent is unclear

Context (Feedback Data):
{context}

Question/Input: {question}

Response:"""),
    ("human", "Please help me with: {question}")
])
