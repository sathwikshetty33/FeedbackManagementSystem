from langchain_core.tools import tool
import pandas as pd
from io import BytesIO
import requests
from typing import Dict, Any, List
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@tool
def feature_analysis_tool(sheet_url: str, features: List[str]) -> Dict[str, Any]:
    """Analyze dataset features directly from Google Sheet.
    
    This tool loads data from a Google Sheet and performs statistical analysis
    on the specified features/columns.
    
    Args:
        sheet_url: URL of the Google Sheet containing the dataset
        features: List of feature/column names to analyze
    
    Returns:
        Dictionary with statistical analysis for each feature including:
        - For numeric columns: mean, median, std_dev, min, max
        - For categorical columns: mode, unique_count, most_frequent
        - Column type information for all columns
    """
    try:
        logger.info(f"Analyzing features {features[:3]}... from sheet: {sheet_url[:50]}...")
        
        # Extract the sheet ID from various Google Sheets URL formats
        sheet_id = None
        
        # Pattern 1: /d/{ID}/edit or /d/{ID}
        match = re.search(r'/d/([a-zA-Z0-9-_]+)', sheet_url)
        if match:
            sheet_id = match.group(1)
        
        # Pattern 2: Direct sheet ID
        elif len(sheet_url) > 20 and '/' not in sheet_url:
            sheet_id = sheet_url
        
        if not sheet_id:
            return {"error": "Could not extract Google Sheet ID from URL"}
        
        logger.info(f"Extracted Sheet ID: {sheet_id}")
        
        # Build clean CSV export URL
        csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
        
        logger.info(f"Fetching data from: {csv_url}")
        
        # Load the data with headers to handle authentication better
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(csv_url, timeout=15, headers=headers)
        response.raise_for_status()
        
        # Check if we got HTML instead of CSV (indicates permission/auth issue)
        if response.headers.get('content-type', '').startswith('text/html'):
            return {
                "error": "Google Sheet is not publicly accessible. Please make sure the sheet is shared with 'Anyone with the link' can view."
            }
        
        df = pd.read_csv(BytesIO(response.content))
        
        logger.info(f"Loaded dataframe with {len(df)} rows and {len(df.columns)} columns")
        logger.info(f"Available columns: {list(df.columns)}")

        results = {}
        
        # Analyze each requested feature
        for col in features:
            # Try exact match first
            if col in df.columns:
                target_col = col
            else:
                # Try case-insensitive partial match
                matches = [c for c in df.columns if col.lower() in c.lower()]
                if matches:
                    target_col = matches[0]
                    logger.info(f"Matched '{col}' to column '{target_col}'")
                else:
                    logger.warning(f"Column '{col}' not found in dataset")
                    results[col] = {"error": "Column not found"}
                    continue

            dtype = str(df[target_col].dtype)
            analysis = {"type": dtype, "column_name": target_col}

            # Numeric column analysis
            if pd.api.types.is_numeric_dtype(df[target_col]):
                clean_data = df[target_col].dropna()
                if len(clean_data) > 0:
                    analysis.update({
                        "mean": round(float(clean_data.mean()), 2),
                        "median": round(float(clean_data.median()), 2),
                        "std_dev": round(float(clean_data.std()), 2),
                        "min": float(clean_data.min()),
                        "max": float(clean_data.max()),
                        "count": int(len(clean_data)),
                        "missing": int(df[target_col].isna().sum())
                    })
                else:
                    analysis["error"] = "No valid numeric data"
                    
            # Categorical/text column analysis
            else:
                clean_data = df[target_col].dropna().astype(str)
                if len(clean_data) > 0:
                    value_counts = clean_data.value_counts()
                    
                    # Get distribution of all values
                    distribution = {}
                    for val, count in value_counts.items():
                        distribution[val] = {
                            "count": int(count),
                            "percentage": round((count / len(clean_data)) * 100, 2)
                        }
                    
                    analysis.update({
                        "mode": clean_data.mode().iloc[0] if not clean_data.mode().empty else None,
                        "unique_count": int(clean_data.nunique()),
                        "most_frequent": value_counts.idxmax() if len(value_counts) > 0 else None,
                        "most_frequent_count": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                        "distribution": distribution,
                        "count": int(len(clean_data)),
                        "missing": int(df[target_col].isna().sum())
                    })
                else:
                    analysis["error"] = "No valid data"

            results[col] = analysis
            logger.info(f"✅ Analyzed column '{target_col}': {analysis.get('type', 'unknown')}")

        # Include overview of all columns
        results["_metadata"] = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "all_columns": list(df.columns),
            "column_types": {col: str(df[col].dtype) for col in df.columns}
        }

        logger.info(f"Successfully analyzed {len([k for k in results.keys() if k != '_metadata'])} features")
        return results

    except requests.exceptions.HTTPError as e:
        error_msg = f"Failed to fetch Google Sheet (HTTP {e.response.status_code}). Make sure the sheet is publicly accessible (shared with 'Anyone with the link')."
        logger.error(error_msg)
        return {"error": error_msg}
    
    except requests.exceptions.RequestException as e:
        error_msg = f"Network error: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}
    
    except pd.errors.ParserError as e:
        error_msg = f"Failed to parse CSV data: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}
    
    except Exception as e:
        error_msg = f"Analysis failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {"error": error_msg}
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from .configs import GroqChatRag
from typing import TypedDict, Literal, List, Dict, Any
import logging

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

config = GroqChatRag()
llm = ChatGroq(
    model=config.LLM_MODEL,
    temperature=config.TEMPERATURE,
    max_tokens=config.MAX_TOKEN,
    groq_api_key=config.GROQ_API_KEY
)

# Define state schema
class GraphState(TypedDict):
    question: str
    context: str
    sheet_url: str
    dataset_description: List[Dict[str, Any]]
    analysis: Dict[str, Any]
    answer: str
    decision: str

def build_tools_graph():
    workflow = StateGraph(GraphState)

    def decision_node(state: GraphState) -> GraphState:
        """Decides if statistical analysis is needed - returns updated state"""
        logger.info("=" * 80)
        logger.info("🔍 DECISION NODE - Starting analysis")
        logger.info("=" * 80)
        
        query = state["question"]
        chunks = state["context"]
        available_columns = [col.get("name", str(col)) for col in state["dataset_description"]]
        
        logger.info(f"📝 Query: {query}")
        logger.info(f"📊 Available columns: {available_columns}")
        logger.info(f"📄 Context length: {len(chunks)} characters")
        
        # Few-shot prompting with clear examples
        prompt = f"""You are a decision agent that determines if a statistical analysis tool is needed.

CRITICAL INFORMATION:
- The context provided is ONLY a small sample (5-10 rows) from a much larger dataset
- You have access to a tool that can analyze the COMPLETE dataset with accurate statistics
- Available columns: {', '.join(available_columns)}

USER QUERY: "{query}"

SAMPLE CONTEXT (INCOMPLETE DATA):
{chunks[:1500]}

FEW-SHOT EXAMPLES:

Example 1:
Query: "Give statistical analysis of student ratings"
Decision: YES (needs complete dataset statistics)
Reason: Query explicitly asks for statistical analysis

Example 2:
Query: "What is the average score for mathematics knowledge?"
Decision: YES (needs complete dataset statistics)
Reason: Requires calculating average across entire dataset, not just sample

Example 3:
Query: "How many responses rated 'Excellent'?"
Decision: YES (needs complete dataset statistics)
Reason: Needs accurate count from complete dataset

Example 4:
Query: "What is machine learning?"
Decision: NO (context sufficient)
Reason: General knowledge question, doesn't need dataset analysis

Example 5:
Query: "Explain the feedback process"
Decision: NO (context sufficient)
Reason: Qualitative question answerable from context

YOUR TURN:
Query: "{query}"

ANALYSIS CHECKLIST:
1. Does query ask for statistics/analysis? (average, count, percentage, distribution, etc.)
2. Does query mention specific columns like: {', '.join(available_columns[:3])}?
3. Would accurate numbers from the COMPLETE dataset improve the answer?
4. Is this asking about data patterns that require seeing all records?

If you answered YES to any of the above, you MUST use the tool.

Decision (answer ONLY with 'yes' or 'no'):"""

        logger.info("🤖 Sending decision prompt to LLM...")
        response = llm.invoke(prompt).content.strip().lower()
        logger.info(f"🎯 LLM Response: '{response}'")
        
        # Parse decision
        if "yes" in response:
            decision = "need_more"
            logger.info("✅ DECISION: USE TOOL - Will fetch complete dataset statistics")
        else:
            decision = "enough"
            logger.info("❌ DECISION: SKIP TOOL - Will use context only")
        
        logger.info("=" * 80)
        
        return {"decision": decision}

    def mcp_node(state: GraphState) -> GraphState:
        """Fetches feature analysis using the tool"""
        logger.info("=" * 80)
        logger.info("🔧 MCP NODE - Calling analysis tool")
        logger.info("=" * 80)
        
        sheet_url = state["sheet_url"]
        features = [f.get("name", str(f)) for f in state["dataset_description"]]
        
        logger.info(f"🌐 Sheet URL: {sheet_url[:50]}...")
        logger.info(f"📊 Features to analyze: {features}")
        
        try:
            # Use the tool to get analysis
            logger.info("⏳ Invoking feature_analysis_tool...")
            analysis = feature_analysis_tool.invoke({
                "sheet_url": sheet_url,
                "features": features
            })
            
            if analysis.get("error"):
                logger.error(f"❌ Tool returned error: {analysis['error']}")
            else:
                logger.info(f"✅ Tool succeeded! Analyzed {len([k for k in analysis.keys() if k != '_metadata'])} features")
                if "_metadata" in analysis:
                    logger.info(f"📈 Total rows in dataset: {analysis['_metadata'].get('total_rows', 'N/A')}")
        
        except Exception as e:
            logger.error(f"❌ Tool invocation failed: {str(e)}")
            analysis = {"error": str(e)}
        
        logger.info("=" * 80)
        
        return {"analysis": analysis}

    def answer_node(state: GraphState) -> GraphState:
        """Produces final answer using LLM"""
        logger.info("=" * 80)
        logger.info("💬 ANSWER NODE - Generating final response")
        logger.info("=" * 80)
        
        query = state["question"]
        chunks = state["context"]
        analysis = state.get("analysis", {})
        decision = state.get("decision", "unknown")
        
        logger.info(f"📝 Query: {query}")
        logger.info(f"🔍 Decision was: {decision}")
        logger.info(f"📊 Has analysis data: {bool(analysis and not analysis.get('error'))}")

        # Build enhanced prompt with analysis
        analysis_text = ""
        used_tool = False
        
        if analysis and not analysis.get("error"):
            used_tool = True
            analysis_text = "\n\n" + "=" * 60 + "\n"
            analysis_text += "📊 COMPLETE DATASET STATISTICAL ANALYSIS\n"
            analysis_text += "=" * 60 + "\n"
            
            # Extract metadata first
            metadata = analysis.get("_metadata", {})
            if metadata:
                analysis_text += f"\n📈 Dataset Overview:\n"
                analysis_text += f"   • Total Records: {metadata.get('total_rows', 'N/A')}\n"
                analysis_text += f"   • Total Columns: {metadata.get('total_columns', 'N/A')}\n\n"
            
            # Add feature-specific analysis
            for feature, stats in analysis.items():
                if feature != "_metadata" and isinstance(stats, dict) and not stats.get("error"):
                    analysis_text += f"\n📊 Column: {feature}\n"
                    analysis_text += f"   Type: {stats.get('type', 'Unknown')}\n"
                    
                    # Numeric statistics
                    if 'mean' in stats:
                        analysis_text += f"   • Mean: {stats.get('mean', 'N/A')}\n"
                        analysis_text += f"   • Median: {stats.get('median', 'N/A')}\n"
                        analysis_text += f"   • Std Dev: {stats.get('std_dev', 'N/A')}\n"
                        analysis_text += f"   • Min: {stats.get('min', 'N/A')}\n"
                        analysis_text += f"   • Max: {stats.get('max', 'N/A')}\n"
                        analysis_text += f"   • Count: {stats.get('count', 'N/A')}\n"
                        analysis_text += f"   • Missing: {stats.get('missing', 'N/A')}\n"
                    
                    # Categorical statistics
                    if 'most_frequent' in stats:
                        analysis_text += f"   • Most Frequent: {stats.get('most_frequent', 'N/A')}\n"
                        analysis_text += f"   • Frequency: {stats.get('most_frequent_count', 'N/A')}\n"
                        analysis_text += f"   • Unique Values: {stats.get('unique_count', 'N/A')}\n"
                        analysis_text += f"   • Mode: {stats.get('mode', 'N/A')}\n"
                        analysis_text += f"   • Count: {stats.get('count', 'N/A')}\n"
                        analysis_text += f"   • Missing: {stats.get('missing', 'N/A')}\n"
            
            logger.info("✅ Statistical analysis included in prompt")
        else:
            logger.info("⚠️  No statistical analysis available, using context only")
        
        prompt = f"""You are a data analysis assistant providing accurate statistical insights.

USER QUESTION: {query}

{"SAMPLE CONTEXT (Limited Data):" if not used_tool else "REFERENCE CONTEXT:"}
{chunks[:2000]}
{analysis_text}

CRITICAL INSTRUCTIONS:
{"1. USE THE STATISTICAL ANALYSIS DATA ABOVE - these are accurate numbers from the COMPLETE dataset" if used_tool else "1. You only have sample data - mention this limitation in your answer"}
2. Be specific with numbers when available
3. If statistical analysis shows different numbers than the sample, USE THE STATISTICAL ANALYSIS
4. Format your response clearly with the actual statistics
5. Don't make up numbers - only use what's provided

Generate your answer now:"""

        logger.info("🤖 Sending answer prompt to LLM...")
        response = llm.invoke(prompt).content.strip()
        logger.info(f"✅ Generated answer ({len(response)} characters)")
        logger.info("=" * 80)
        
        return {"answer": response}

    # Add nodes
    workflow.add_node("decision", decision_node)
    workflow.add_node("mcp_analysis", mcp_node)
    workflow.add_node("answer", answer_node)

    # Define routing function
    def route_decision(state: GraphState) -> Literal["answer", "mcp_analysis"]:
        """Routes based on decision in state"""
        decision = state.get("decision", "need_more")
        
        logger.info("🔀 ROUTING DECISION:")
        logger.info(f"   Decision value: {decision}")
        
        if decision == "enough":
            logger.info("   → Route: Skip tool, go directly to ANSWER")
            return "answer"
        else:
            logger.info("   → Route: Call MCP_ANALYSIS tool first")
            return "mcp_analysis"

    # Add conditional edges
    workflow.add_conditional_edges(
        "decision",
        route_decision,
        {
            "answer": "answer",
            "mcp_analysis": "mcp_analysis"
        }
    )
    
    workflow.add_edge("mcp_analysis", "answer")
    workflow.add_edge("answer", END)

    workflow.set_entry_point("decision")

    logger.info("✅ LangGraph workflow compiled successfully")
    
    return workflow.compile()