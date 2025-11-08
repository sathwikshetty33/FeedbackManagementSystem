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


@tool
def filter_rows_tool(sheet_url: str, conditions: Dict[str, Any]) -> Dict[str, Any]:
    """Filter and retrieve rows from Google Sheet based on multiple conditions.
    
    This tool loads data from a Google Sheet and filters rows that match
    all the specified conditions (AND logic).
    
    Args:
        sheet_url: URL of the Google Sheet containing the dataset
        conditions: Dictionary mapping column names to their expected values
                   Example: {"Status": "Active", "Score": 85, "Department": "Engineering"}
                   Supports exact matches for strings and numbers
    
    Returns:
        Dictionary containing:
        - filtered_rows: List of dictionaries representing matching rows
        - total_matches: Number of rows that matched the conditions
        - total_rows: Total number of rows in dataset
        - conditions_applied: The conditions that were used for filtering
        - columns: List of all column names
    """
    try:
        logger.info(f"Filtering rows with conditions: {conditions}")
        logger.info(f"From sheet: {sheet_url[:50]}...")
        
        # Extract the sheet ID
        sheet_id = None
        match = re.search(r'/d/([a-zA-Z0-9-_]+)', sheet_url)
        if match:
            sheet_id = match.group(1)
        elif len(sheet_url) > 20 and '/' not in sheet_url:
            sheet_id = sheet_url
        
        if not sheet_id:
            return {"error": "Could not extract Google Sheet ID from URL"}
        
        logger.info(f"Extracted Sheet ID: {sheet_id}")
        
        # Build CSV export URL
        csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
        
        # Fetch data
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(csv_url, timeout=15, headers=headers)
        response.raise_for_status()
        
        if response.headers.get('content-type', '').startswith('text/html'):
            return {
                "error": "Google Sheet is not publicly accessible. Please make sure the sheet is shared with 'Anyone with the link' can view."
            }
        
        df = pd.read_csv(BytesIO(response.content))
        
        logger.info(f"Loaded dataframe with {len(df)} rows and {len(df.columns)} columns")
        logger.info(f"Available columns: {list(df.columns)}")
        
        # Start with all rows
        filtered_df = df.copy()
        applied_conditions = {}
        
        # Apply each condition
        for col_name, expected_value in conditions.items():
            # Try exact match first
            if col_name in df.columns:
                target_col = col_name
            else:
                # Try case-insensitive partial match
                matches = [c for c in df.columns if col_name.lower() in c.lower()]
                if matches:
                    target_col = matches[0]
                    logger.info(f"Matched '{col_name}' to column '{target_col}'")
                else:
                    logger.warning(f"Column '{col_name}' not found in dataset")
                    return {
                        "error": f"Column '{col_name}' not found",
                        "available_columns": list(df.columns)
                    }
            
            # Apply filter based on data type
            if pd.api.types.is_numeric_dtype(filtered_df[target_col]):
                # Numeric comparison
                try:
                    numeric_value = float(expected_value)
                    filtered_df = filtered_df[filtered_df[target_col] == numeric_value]
                    applied_conditions[target_col] = numeric_value
                    logger.info(f"Applied numeric filter: {target_col} == {numeric_value}")
                except (ValueError, TypeError):
                    logger.warning(f"Could not convert '{expected_value}' to numeric for column '{target_col}'")
                    return {
                        "error": f"Value '{expected_value}' cannot be compared to numeric column '{target_col}'"
                    }
            else:
                # String comparison (case-insensitive)
                str_value = str(expected_value).strip()
                filtered_df = filtered_df[
                    filtered_df[target_col].astype(str).str.strip().str.lower() == str_value.lower()
                ]
                applied_conditions[target_col] = str_value
                logger.info(f"Applied string filter: {target_col} == '{str_value}' (case-insensitive)")
        
        # Convert filtered results to list of dictionaries
        filtered_rows = filtered_df.to_dict('records')
        
        # Clean up NaN values in the output
        for row in filtered_rows:
            for key, value in row.items():
                if pd.isna(value):
                    row[key] = None
        
        result = {
            "filtered_rows": filtered_rows,
            "total_matches": len(filtered_df),
            "total_rows": len(df),
            "conditions_applied": applied_conditions,
            "columns": list(df.columns),
            "match_percentage": round((len(filtered_df) / len(df) * 100), 2) if len(df) > 0 else 0
        }
        
        logger.info(f"✅ Filter complete: {len(filtered_df)} matches out of {len(df)} total rows")
        
        return result

    except requests.exceptions.HTTPError as e:
        error_msg = f"Failed to fetch Google Sheet (HTTP {e.response.status_code}). Make sure the sheet is publicly accessible."
        logger.error(error_msg)
        return {"error": error_msg}
    
    except requests.exceptions.RequestException as e:
        error_msg = f"Network error: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}
    
    except Exception as e:
        error_msg = f"Row filtering failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {"error": error_msg}


# =============================================================================
# LANGGRAPH WORKFLOW WITH BOTH TOOLS
# =============================================================================

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from typing import TypedDict, Literal, List, Dict, Any
import logging

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from .configs import GroqChatRag
config = GroqChatRag()
llm = ChatGroq(
    model=config.LLM_MODEL,
    temperature=config.TEMPERATURE,
    max_tokens=config.MAX_TOKEN,
    groq_api_key=config.GROQ_API_KEY
)

# Enhanced state schema with filter support
class GraphState(TypedDict):
    question: str
    context: str
    sheet_url: str
    dataset_description: List[Dict[str, Any]]
    analysis: Dict[str, Any]
    filtered_data: Dict[str, Any]
    answer: str
    decision: str
    needs_filtering: bool
    filter_conditions: Dict[str, Any]

def build_tools_graph():
    workflow = StateGraph(GraphState)

    def decision_node(state: GraphState) -> GraphState:
        """Decides if statistical analysis and/or row filtering is needed"""
        logger.info("=" * 80)
        logger.info("🔍 DECISION NODE - Starting analysis")
        logger.info("=" * 80)
        
        query = state["question"]
        chunks = state["context"]
        available_columns = [col.get("name", str(col)) for col in state["dataset_description"]]
        
        logger.info(f"📝 Query: {query}")
        logger.info(f"📊 Available columns: {available_columns}")
        
        # Decision prompt for both tools
        prompt = f"""You are a decision agent that determines which tools are needed to answer a query.

AVAILABLE TOOLS:
1. Statistical Analysis Tool - Analyzes entire dataset for statistics (mean, median, counts, distributions)
2. Row Filter Tool - Retrieves specific rows matching conditions

AVAILABLE COLUMNS: {', '.join(available_columns)}

USER QUERY: "{query}"

SAMPLE CONTEXT:
{chunks[:1500]}

DECISION CRITERIA:

Use Statistical Analysis if query asks for:
- Statistics (average, mean, median, std dev, min, max)
- Counts, percentages, distributions
- Overall patterns or trends
Examples: "What's the average score?", "How many students rated Excellent?"

Use Row Filter if query asks for:
- Specific rows matching criteria
- Details about records with certain conditions
- Information filtered by specific values
Examples: "Show students who rated Excellent", "Get records where department is Engineering"

YOUR TURN:
Query: "{query}"

Respond in this EXACT format:
NEEDS_ANALYSIS: yes/no
NEEDS_FILTERING: yes/no
FILTER_CONDITIONS: {{column1: value1, column2: value2}} (if filtering needed, otherwise empty)

Example responses:
Query: "What's the average rating?"
NEEDS_ANALYSIS: yes
NEEDS_FILTERING: no
FILTER_CONDITIONS: {{}}

Query: "Show me students who rated 'Excellent' in Mathematics"
NEEDS_ANALYSIS: no
NEEDS_FILTERING: yes
FILTER_CONDITIONS: {{"Subject": "Mathematics", "Rating": "Excellent"}}

Query: "What's the average score for Excellent ratings?"
NEEDS_ANALYSIS: yes
NEEDS_FILTERING: yes
FILTER_CONDITIONS: {{"Rating": "Excellent"}}

Your response:"""

        logger.info("🤖 Sending decision prompt to LLM...")
        response = llm.invoke(prompt).content.strip()
        logger.info(f"🎯 LLM Response:\n{response}")
        
        # Parse response
        needs_analysis = "yes" in response.split("NEEDS_ANALYSIS:")[1].split("\n")[0].lower()
        needs_filtering = "yes" in response.split("NEEDS_FILTERING:")[1].split("\n")[0].lower()
        
        # Extract filter conditions
        filter_conditions = {}
        try:
            if "FILTER_CONDITIONS:" in response:
                conditions_str = response.split("FILTER_CONDITIONS:")[1].strip()
                if conditions_str and conditions_str != "{}":
                    # Simple parsing of dictionary format
                    import ast
                    filter_conditions = ast.literal_eval(conditions_str)
        except Exception as e:
            logger.warning(f"Could not parse filter conditions: {e}")
        
        # Set decision based on what's needed
        if needs_analysis and needs_filtering:
            decision = "need_both"
        elif needs_analysis:
            decision = "need_analysis"
        elif needs_filtering:
            decision = "need_filtering"
        else:
            decision = "enough"
        
        logger.info(f"✅ DECISION: {decision}")
        logger.info(f"   Analysis needed: {needs_analysis}")
        logger.info(f"   Filtering needed: {needs_filtering}")
        logger.info(f"   Filter conditions: {filter_conditions}")
        logger.info("=" * 80)
        
        return {
            "decision": decision,
            "needs_filtering": needs_filtering,
            "filter_conditions": filter_conditions
        }

    def mcp_analysis_node(state: GraphState) -> GraphState:
        """Fetches feature analysis using the tool"""
        logger.info("=" * 80)
        logger.info("📊 MCP ANALYSIS NODE - Calling analysis tool")
        logger.info("=" * 80)
        
        sheet_url = state["sheet_url"]
        features = [f.get("name", str(f)) for f in state["dataset_description"]]
        
        logger.info(f"🌐 Sheet URL: {sheet_url[:50]}...")
        logger.info(f"📊 Features to analyze: {features}")
        
        try:
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

    def filter_rows_node(state: GraphState) -> GraphState:
        """Filters rows based on conditions"""
        logger.info("=" * 80)
        logger.info("🔍 FILTER ROWS NODE - Calling filter tool")
        logger.info("=" * 80)
        
        sheet_url = state["sheet_url"]
        conditions = state.get("filter_conditions", {})
        
        logger.info(f"🌐 Sheet URL: {sheet_url[:50]}...")
        logger.info(f"🔧 Filter conditions: {conditions}")
        
        if not conditions:
            logger.warning("⚠️  No filter conditions provided")
            return {"filtered_data": {"error": "No filter conditions specified"}}
        
        try:
            logger.info("⏳ Invoking filter_rows_tool...")
            filtered_data = filter_rows_tool.invoke({
                "sheet_url": sheet_url,
                "conditions": conditions
            })
            
            if filtered_data.get("error"):
                logger.error(f"❌ Tool returned error: {filtered_data['error']}")
            else:
                logger.info(f"✅ Tool succeeded! Found {filtered_data.get('total_matches', 0)} matching rows")
        
        except Exception as e:
            logger.error(f"❌ Tool invocation failed: {str(e)}")
            filtered_data = {"error": str(e)}
        
        logger.info("=" * 80)
        
        return {"filtered_data": filtered_data}

    def answer_node(state: GraphState) -> GraphState:
        """Produces final answer using LLM"""
        logger.info("=" * 80)
        logger.info("💬 ANSWER NODE - Generating final response")
        logger.info("=" * 80)
        
        query = state["question"]
        chunks = state["context"]
        analysis = state.get("analysis", {})
        filtered_data = state.get("filtered_data", {})
        decision = state.get("decision", "unknown")
        
        logger.info(f"📝 Query: {query}")
        logger.info(f"🔍 Decision was: {decision}")
        logger.info(f"📊 Has analysis data: {bool(analysis and not analysis.get('error'))}")
        logger.info(f"🔍 Has filtered data: {bool(filtered_data and not filtered_data.get('error'))}")

        # Build enhanced prompt
        additional_context = ""
        
        # Add statistical analysis
        if analysis and not analysis.get("error"):
            additional_context += "\n\n" + "=" * 60 + "\n"
            additional_context += "📊 COMPLETE DATASET STATISTICAL ANALYSIS\n"
            additional_context += "=" * 60 + "\n"
            
            metadata = analysis.get("_metadata", {})
            if metadata:
                additional_context += f"\n📈 Dataset Overview:\n"
                additional_context += f"   • Total Records: {metadata.get('total_rows', 'N/A')}\n"
                additional_context += f"   • Total Columns: {metadata.get('total_columns', 'N/A')}\n\n"
            
            for feature, stats in analysis.items():
                if feature != "_metadata" and isinstance(stats, dict) and not stats.get("error"):
                    additional_context += f"\n📊 Column: {feature}\n"
                    additional_context += f"   Type: {stats.get('type', 'Unknown')}\n"
                    
                    if 'mean' in stats:
                        additional_context += f"   • Mean: {stats.get('mean')}\n"
                        additional_context += f"   • Median: {stats.get('median')}\n"
                        additional_context += f"   • Std Dev: {stats.get('std_dev')}\n"
                        additional_context += f"   • Range: {stats.get('min')} to {stats.get('max')}\n"
                    
                    if 'most_frequent' in stats:
                        additional_context += f"   • Most Frequent: {stats.get('most_frequent')} ({stats.get('most_frequent_count')} times)\n"
                        additional_context += f"   • Unique Values: {stats.get('unique_count')}\n"
            
            logger.info("✅ Statistical analysis included in prompt")
        
        # Add filtered rows
        if filtered_data and not filtered_data.get("error"):
            additional_context += "\n\n" + "=" * 60 + "\n"
            additional_context += "🔍 FILTERED ROWS (Matching Conditions)\n"
            additional_context += "=" * 60 + "\n"
            additional_context += f"Conditions Applied: {filtered_data.get('conditions_applied', {})}\n"
            additional_context += f"Total Matches: {filtered_data.get('total_matches', 0)} out of {filtered_data.get('total_rows', 0)} rows\n"
            additional_context += f"Match Percentage: {filtered_data.get('match_percentage', 0)}%\n\n"
            
            rows = filtered_data.get('filtered_rows', [])
            if rows:
                additional_context += "Matching Records:\n"
                for idx, row in enumerate(rows[:50], 1):  # Limit to first 50 rows
                    additional_context += f"\nRecord {idx}:\n"
                    for key, value in row.items():
                        additional_context += f"   • {key}: {value}\n"
                
                if len(rows) > 50:
                    additional_context += f"\n... and {len(rows) - 50} more records\n"
            
            logger.info("✅ Filtered data included in prompt")
        
        prompt = f"""You are a data analysis assistant providing accurate insights.

USER QUESTION: {query}

REFERENCE CONTEXT:
{chunks[:2000]}
{additional_context}

INSTRUCTIONS:
1. Use the statistical analysis and filtered data provided above
2. Be specific with numbers and facts
3. Format your response clearly
4. If you have filtered rows, reference them specifically
5. Don't make up information - only use what's provided

Generate your answer now:"""

        logger.info("🤖 Sending answer prompt to LLM...")
        response = llm.invoke(prompt).content.strip()
        logger.info(f"✅ Generated answer ({len(response)} characters)")
        logger.info("=" * 80)
        
        return {"answer": response}

    # Add all nodes
    workflow.add_node("decision", decision_node)
    workflow.add_node("mcp_analysis", mcp_analysis_node)
    workflow.add_node("filter_rows", filter_rows_node)
    workflow.add_node("answer", answer_node)

    # Routing function
    def route_decision(state: GraphState) -> Literal["answer", "mcp_analysis", "filter_rows"]:
        """Routes based on decision in state"""
        decision = state.get("decision", "enough")
        
        logger.info("🔀 ROUTING DECISION:")
        logger.info(f"   Decision value: {decision}")
        
        if decision == "enough":
            logger.info("   → Route: Go directly to ANSWER")
            return "answer"
        elif decision == "need_analysis":
            logger.info("   → Route: Call MCP_ANALYSIS first")
            return "mcp_analysis"
        elif decision == "need_filtering":
            logger.info("   → Route: Call FILTER_ROWS first")
            return "filter_rows"
        else:  # need_both
            logger.info("   → Route: Call MCP_ANALYSIS first (will chain to filtering)")
            return "mcp_analysis"

    # Add conditional edges from decision node
    workflow.add_conditional_edges(
        "decision",
        route_decision,
        {
            "answer": "answer",
            "mcp_analysis": "mcp_analysis",
            "filter_rows": "filter_rows"
        }
    )
    
    # Chain tools when both are needed
    def route_after_analysis(state: GraphState) -> Literal["answer", "filter_rows"]:
        """After analysis, check if filtering is also needed"""
        if state.get("needs_filtering", False):
            logger.info("   → After analysis: Going to FILTER_ROWS")
            return "filter_rows"
        else:
            logger.info("   → After analysis: Going to ANSWER")
            return "answer"
    
    workflow.add_conditional_edges(
        "mcp_analysis",
        route_after_analysis,
        {
            "answer": "answer",
            "filter_rows": "filter_rows"
        }
    )
    
    # After filtering, always go to answer
    workflow.add_edge("filter_rows", "answer")
    workflow.add_edge("answer", END)

    workflow.set_entry_point("decision")

    logger.info("✅ Enhanced LangGraph workflow compiled successfully")
    
    return workflow.compile()

