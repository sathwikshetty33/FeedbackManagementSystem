from dotenv import load_dotenv

load_dotenv()
import os
import json
import logging
import requests
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.authentication import TokenAuthentication
from rest_framework.permissions import IsAuthenticated
from rest_framework import status
from home.models import Event
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from functools import lru_cache

# Configure logging properly
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Output to console
        logging.FileHandler('feedback_insights.log')  # Also save to a file
    ]
)
logger = logging.getLogger(__name__)

# Verify logging is working
logger.info("Logging system initialized")

# Cache for ollama responses to avoid redundant API calls
@lru_cache(maxsize=32)
def cached_ollama_request(prompt_hash):
    """Cache wrapper for Ollama API requests"""
    return None  # Just a placeholder, the actual function will cache based on the hash

class GenerateInsightsView(APIView):
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize API endpoint from environment
        self.ollama_api_url = os.environ.get('OLLAMA_API_URL', 'http://localhost:11434/api/generate')
        # Initialize vectorizer once for reuse
        self.vectorizer = TfidfVectorizer(stop_words='english')
        # Configure chunking parameters
        self.chunk_size = int(os.environ.get('INSIGHT_CHUNK_SIZE', '10'))
        # Configure concurrency
        self.max_workers = min(int(os.environ.get('MAX_OLLAMA_WORKERS', '3')), 5)  # Limit to avoid overloading
        # Configure token limits to prevent timeouts
        self.max_tokens_chunk = int(os.environ.get('MAX_TOKENS_CHUNK', '800'))
        self.max_tokens_synthesis = int(os.environ.get('MAX_TOKENS_SYNTHESIS', '1500'))
        # Set timeout for Ollama requests
        self.request_timeout = int(os.environ.get('OLLAMA_TIMEOUT', '60'))
        
        # Log configuration
        logger.info(f"Initialized with: Ollama API URL={self.ollama_api_url}, "
                   f"Max workers={self.max_workers}, Chunk size={self.chunk_size}, "
                   f"Request timeout={self.request_timeout}s")
    
    def extract_data_from_sheet(self, worksheet_url):
        """
        Extract data from Google Sheets with improved error handling and caching
        """
        try:
            logger.info(f"Extracting data from: {worksheet_url}")
            
            # Convert Google Sheets URL to export format
            if "spreadsheets/d/" in worksheet_url:
                sheet_id = worksheet_url.split("spreadsheets/d/")[1].split("/")[0]
                export_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
                
                # Add request timeout and error handling
                response = requests.get(export_url, timeout=10)
                if response.status_code != 200:
                    logger.error(f"Failed to fetch Google Sheet: HTTP {response.status_code}")
                    return None, None
                
                # Use StringIO to avoid unnecessary file operations
                from io import StringIO
                df = pd.read_csv(StringIO(response.text))
                
                # Early check for empty dataframe
                if df.empty:
                    logger.warning("Sheet contains no data")
                    return None, None
                    
                data = df.to_dict(orient='records')
                logger.info(f"Successfully extracted {len(data)} records from sheet")
                return df, data
            else:
                logger.error(f"Invalid Google Sheets URL format: {worksheet_url}")
                return None, None
        except requests.exceptions.Timeout:
            logger.error(f"Timeout extracting data from sheet: {worksheet_url}")
            return None, None
        except Exception as e:
            logger.error(f"Error extracting data from sheet: {str(e)}", exc_info=True)
            return None, None
    
    def preprocess_feedback_data(self, df):
        """
        Optimized preprocessing with better heuristics to identify relevant columns
        """
        try:
            logger.info("Starting feedback data preprocessing")
            
            # Create a copy to avoid modifying original data
            processed_df = df.copy()
            
            # Identify potential feedback columns using heuristics
            potential_feedback_cols = []
            
            # List of common PII/irrelevant columns to exclude
            common_exclude_columns = [
                'name', 'fullname', 'first name', 'last name', 'firstname', 'lastname', 
                'usn', 'registration', 'student id', 'studentid', 'id', 'email', 'phone', 
                'mobile', 'address', 'timestamp', 'submitted at', 'submissiondate',
                'ip address', 'age', 'roll no', 'dob', 'date of birth'
            ]
            
            # Case-insensitive column filtering - use regex for faster matching
            exclude_pattern = '|'.join(common_exclude_columns)
            columns_to_drop = [col for col in processed_df.columns 
                              if re.search(exclude_pattern, col, re.IGNORECASE)]
            
            # Drop identified columns
            if columns_to_drop:
                logger.info(f"Dropping {len(columns_to_drop)} irrelevant columns")
                processed_df = processed_df.drop(columns=columns_to_drop, errors='ignore')
            
            # Find columns with text data that might contain feedback
            for col in processed_df.columns:
                if processed_df[col].dtype == 'object':  # If column is text/categorical
                    # Check if column likely contains feedback (has longer text in some rows)
                    if processed_df[col].str.len().max() > 20:  # Threshold for feedback text
                        potential_feedback_cols.append(col)
                    processed_df[col] = processed_df[col].fillna('No response')
            
            # For numeric columns, fill with median more efficiently
            numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                median_val = processed_df[col].median()
                processed_df[col] = processed_df[col].fillna(median_val)
            
            # Add metadata about which columns might contain valuable feedback
            if potential_feedback_cols:
                logger.info(f"Identified {len(potential_feedback_cols)} potential feedback columns: {potential_feedback_cols}")
            
            return processed_df
        except Exception as e:
            logger.error(f"Error preprocessing feedback data: {str(e)}", exc_info=True)
            return df  # Return original data if preprocessing fails
    
    def generate_embeddings(self, texts):
        """
        Generate efficient TF-IDF embeddings for text data
        """
        # Skip empty texts
        valid_texts = [t if isinstance(t, str) and t.strip() else "Empty response" for t in texts]
        return self.vectorizer.fit_transform(valid_texts)
    
    def identify_feedback_clusters(self, df):
        """
        New method to identify clusters of similar feedback to optimize processing
        """
        try:
            # Find text columns that might contain feedback
            text_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            # Look for columns with longer text that likely contain feedback
            feedback_cols = []
            for col in text_cols:
                # Check if column has substantial text responses
                if df[col].str.len().mean() > 15:  # Threshold for average text length
                    feedback_cols.append(col)
            
            if not feedback_cols:
                logger.warning("No substantial feedback columns identified")
                # If no clear feedback columns, use all text columns
                feedback_cols = text_cols[:3]  # Limit to first 3 to avoid too much noise
            
            logger.info(f"Using {len(feedback_cols)} columns for feedback analysis: {feedback_cols}")
            
            # Combine text from feedback columns
            combined_text = df[feedback_cols].apply(
                lambda x: ' '.join(str(val) for val in x if pd.notna(val)), 
                axis=1
            )
            
            # Generate embeddings
            embeddings = self.generate_embeddings(combined_text)
            
            # Calculate similarity matrix
            similarity_matrix = cosine_similarity(embeddings)
            
            # Simple clustering by similarity threshold
            clusters = []
            used_indices = set()
            
            for i in range(len(df)):
                if i in used_indices:
                    continue
                    
                # Find similar feedback
                cluster = [i]
                used_indices.add(i)
                
                for j in range(i+1, len(df)):
                    if j not in used_indices and similarity_matrix[i, j] > 0.6:  # Similarity threshold
                        cluster.append(j)
                        used_indices.add(j)
                
                clusters.append(cluster)
            
            logger.info(f"Identified {len(clusters)} feedback clusters")
            return clusters, feedback_cols
            
        except Exception as e:
            logger.error(f"Error identifying feedback clusters: {str(e)}", exc_info=True)
            # Fall back to simple chunking if clustering fails
            return None, None
    
    def chunk_data(self, df, chunk_size=None):
        """
        Improved chunking that considers semantic similarity
        """
        chunk_size = chunk_size or self.chunk_size
        
        # Try to identify feedback clusters first
        clusters, feedback_cols = self.identify_feedback_clusters(df)
        
        if clusters and feedback_cols:
            # Create chunks from clusters
            chunks = []
            for cluster_indices in clusters:
                chunks.append(df.iloc[cluster_indices])
                
            # Break down any chunks that are too large
            final_chunks = []
            for chunk in chunks:
                if len(chunk) > chunk_size * 2:  # If chunk is way too big
                    # Break it down further into smaller chunks
                    for i in range(0, len(chunk), chunk_size):
                        end_idx = min(i + chunk_size, len(chunk))
                        final_chunks.append(chunk.iloc[i:end_idx])
                else:
                    final_chunks.append(chunk)
                    
            logger.info(f"Created {len(final_chunks)} semantically grouped chunks")
            return final_chunks, feedback_cols
        else:
            # Fall back to simple chunking by size
            num_rows = len(df)
            chunks = []
            
            for i in range(0, num_rows, chunk_size):
                end_idx = min(i + chunk_size, num_rows)
                chunks.append(df.iloc[i:end_idx])
                
            logger.info(f"Created {len(chunks)} sequential chunks of size {chunk_size}")
            return chunks, None
    
    def create_prompt_for_chunk(self, chunk_df, feedback_cols=None):
        """
        Create an optimized prompt focusing on the most relevant data
        """
        # Convert chunk to a more compact representation
        if feedback_cols:
            # If we know which columns contain feedback, focus on those
            focused_data = chunk_df[feedback_cols].to_dict(orient='records')
            
            # Add a sample of other columns for context (first 3 rows only)
            other_cols = [col for col in chunk_df.columns if col not in feedback_cols]
            if other_cols:
                context_sample = chunk_df[other_cols].head(3).to_dict(orient='records')
                # Trim the data to save tokens
                data_str = json.dumps({
                    "feedback_data": focused_data,
                    "context_sample": context_sample,
                    "total_responses": len(chunk_df)
                }, ensure_ascii=False)
            else:
                data_str = json.dumps(focused_data, ensure_ascii=False)
        else:
            # No specific feedback columns identified, use the whole chunk
            data_str = json.dumps(chunk_df.to_dict(orient='records'), ensure_ascii=False)
        
        # Create a focused prompt with clear instructions
        prompt = f"""
        Analyze this chunk of feedback data:
        
        {data_str}
        
        Extract only the key insights and themes. Focus on:
        1. Main sentiment (positive/negative/neutral)
        2. Key issues or concerns mentioned
        3. Specific strengths highlighted
        4. Notable suggestions for improvement
        5. Any unique or standout feedback
        
        Be concise and focus only on extracting factual insights. Format as JSON.
        """
        
        return prompt
    
    def request_with_retry(self, prompt, max_tokens, temperature=0.3, retries=2):
        """
        Make API request with retry logic for resilience
        """
        for attempt in range(retries + 1):
            try:
                headers = {"Content-Type": "application/json"}
                
                payload = {
                    "model": "llama3:8b",
                    "prompt": prompt,
                    "stream": False,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
                
                # Create a hash of the prompt for caching
                prompt_hash = hash(prompt + str(max_tokens) + str(temperature))
                cached_result = cached_ollama_request(prompt_hash)
                if cached_result:
                    return cached_result
                
                logger.info(f"Sending request to Ollama API (attempt {attempt+1})")
                
                response = requests.post(
                    self.ollama_api_url,
                    headers=headers,
                    json=payload,
                    timeout=self.request_timeout
                )
                
                if response.status_code == 200:
                    response_data = response.json()
                    result = response_data.get('response', '')
                    
                    # Cache the successful result
                    cached_ollama_request.cache_clear()  # Clear old cache entries
                    cached_ollama_request.__wrapped__ = lambda x: result if x == prompt_hash else None
                    
                    # Log some stats about the response
                    result_length = len(result)
                    logger.info(f"Received response ({result_length} chars) from Ollama API")
                    
                    return result
                else:
                    error_text = response.text[:200]  # Limit error text to prevent huge logs
                    logger.warning(f"API request failed (attempt {attempt+1}): HTTP {response.status_code} - {error_text}")
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Request timeout (attempt {attempt+1}) - consider increasing the timeout value")
            except requests.exceptions.ConnectionError:
                logger.warning(f"Connection error (attempt {attempt+1}) - Ollama service may have crashed or been stopped")
            except Exception as e:
                logger.warning(f"Request error (attempt {attempt+1}): {str(e)}")
                
            # Only sleep if we're going to retry
            if attempt < retries:
                import time
                time.sleep(2 * (attempt + 1))  # Exponential backoff
                
        return None  # All retries failed
    
    def extract_key_insights_from_chunk(self, chunk_df, feedback_cols=None):
        """
        Extract key insights from a single chunk with optimized prompting
        """
        try:
            chunk_size = len(chunk_df)
            logger.info(f"Extracting insights from chunk with {chunk_size} rows")
            
            # Create focused prompt
            prompt = self.create_prompt_for_chunk(chunk_df, feedback_cols)
            
            # Adjust token limit based on chunk size
            token_limit = min(self.max_tokens_chunk, 600 + (chunk_size * 20))
            
            # Make request with retry
            response_text = self.request_with_retry(prompt, token_limit)
            
            if response_text:
                # Try to parse as JSON
                json_content = self._extract_json_from_text(response_text)
                if json_content:
                    try:
                        return json.loads(json_content)
                    except json.JSONDecodeError:
                        logger.warning("Failed to parse response as JSON")
                        return {"raw_insights": response_text[:1000]}  # Truncate if too long
                else:
                    return {"raw_insights": response_text[:1000]}  # Truncate if too long
            else:
                logger.error("Failed to get response from Ollama API")
                return {"error": "API request failed after retries"}
                
        except Exception as e:
            logger.error(f"Error extracting insights from chunk: {str(e)}", exc_info=True)
            return {"error": str(e)}
    
    def _extract_json_from_text(self, text):
        """
        Improved JSON extraction with better regex patterns
        """
        # Try to extract JSON from markdown code blocks
        json_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
        json_matches = re.findall(json_pattern, text)
        
        if json_matches:
            return json_matches[0].strip()
        
        # Try to extract JSON between curly braces
        if "{" in text and "}" in text:
            # Find the first opening brace and last closing brace
            start = text.find("{")
            end = text.rfind("}") + 1
            if start < end:
                return text[start:end]
        
        return text  # Return original if no JSON pattern found
    
    def process_chunks_parallel(self, chunks, feedback_cols=None):
        """
        Process chunks in parallel for better performance
        """
        chunk_insights = []
        failed_chunks = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all chunks for processing
            future_to_chunk = {
                executor.submit(self.extract_key_insights_from_chunk, chunk, feedback_cols): i 
                for i, chunk in enumerate(chunks)
            }
            
            # Process results as they complete
            for future in as_completed(future_to_chunk):
                chunk_idx = future_to_chunk[future]
                try:
                    insight = future.result()
                    if "error" not in insight:
                        logger.info(f"Successfully processed chunk {chunk_idx+1}/{len(chunks)}")
                        chunk_insights.append(insight)
                    else:
                        logger.warning(f"Failed to process chunk {chunk_idx+1}: {insight.get('error')}")
                        failed_chunks += 1
                except Exception as e:
                    logger.error(f"Exception processing chunk {chunk_idx+1}: {str(e)}")
                    failed_chunks += 1
        
        logger.info(f"Completed chunk processing: {len(chunk_insights)} successful, {failed_chunks} failed")
        return chunk_insights
    
    def synthesize_insights(self, chunk_insights):
        """
        Synthesize chunk insights with improved prompt
        """
        try:
            logger.info(f"Synthesizing insights from {len(chunk_insights)} chunks")
            
            # Create a more compact representation for synthesis
            compact_insights = []
            for i, insight in enumerate(chunk_insights):
                # Remove raw_insights fields to save tokens
                compact_version = {k: v for k, v in insight.items() if k != 'raw_insights'}
                # Add index for reference
                compact_version['chunk_index'] = i + 1
                compact_insights.append(compact_version)
            
            # Create synthesis prompt
            prompt = f"""
            Synthesize these insights from {len(chunk_insights)} chunks of feedback data:
            
            {json.dumps(compact_insights, ensure_ascii=False)}
            
            Create a comprehensive analysis with:
            1. Overall sentiment breakdown (positive/negative/neutral percentages)
            2. Top 3-5 key issues needing improvement (with specific details)
            3. Top 3-5 strengths and positive aspects (with specific details)
            4. Clear patterns or trends across the feedback
            5. Actionable recommendations based on the feedback
            
            Format your response as a structured JSON with these sections.
            """
            
            # Make request with retry
            response_text = self.request_with_retry(
                prompt, 
                self.max_tokens_synthesis,
                temperature=0.5,  # Lower temperature for more consistent results
                retries=3  # More retries for this critical step
            )
            
            if response_text:
                # Try to parse as JSON
                json_content = self._extract_json_from_text(response_text)
                if json_content:
                    try:
                        return json.loads(json_content)
                    except json.JSONDecodeError:
                        logger.warning("Failed to parse synthesis response as JSON")
                        return {"raw_insights": response_text[:2000]}  # Truncate if too long
                else:
                    return {"raw_insights": response_text[:2000]}  # Truncate if too long
            else:
                logger.error("Failed to get synthesis response from Ollama API")
                return {"error": "API synthesis request failed after retries"}
                
        except Exception as e:
            logger.error(f"Error synthesizing insights: {str(e)}", exc_info=True)
            return {"error": str(e)}
    
    def generate_insights_with_rag(self, df):
        """
        Generate insights using improved RAG approach
        """
        try:
            logger.info(f"Starting RAG insights generation for dataset with {len(df)} rows")
            
            # 1. Preprocess data
            processed_df = self.preprocess_feedback_data(df)
            
            # 2. Split data into semantically meaningful chunks
            chunks, feedback_cols = self.chunk_data(processed_df)
            
            # 3. Process each chunk in parallel
            chunk_insights = self.process_chunks_parallel(chunks, feedback_cols)
            
            # 4. Handle the case where no insights were generated
            if not chunk_insights:
                logger.error("Failed to extract insights from any data chunks")
                
                # Fallback: Try with a single chunk containing sample rows
                logger.info("Attempting fallback with reduced dataset")
                sample_size = min(25, len(processed_df))
                sample_df = processed_df.sample(n=sample_size) if len(processed_df) > sample_size else processed_df
                
                insight = self.extract_key_insights_from_chunk(sample_df)
                if "error" not in insight:
                    return insight  # Return single chunk insight as final result
                
                return {"error": "Failed to extract insights even with fallback method"}
            
            # 5. For very small datasets, skip synthesis
            if len(chunk_insights) == 1:
                logger.info("Single chunk processed, skipping synthesis")
                return chunk_insights[0]
            
            # 6. Synthesize insights
            final_insights = self.synthesize_insights(chunk_insights)
            
            return final_insights
            
        except Exception as e:
            logger.error(f"Error in RAG insights generation: {str(e)}", exc_info=True)
            return {"error": str(e)}
    
    def post(self, request, *args, **kwargs):
        logger.info("Received feedback analysis request")
        
        event_id = request.data.get('event_id')
        
        if not event_id:
            logger.error("Request missing required 'event_id' parameter")
            return Response({"error": "Event ID is required"}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            # Check if model is available before proceeding
            try:
                # Extract base URL (remove endpoint path)
                base_url = self.ollama_api_url.split('/api/')[0]
                health_url = f"{base_url}/api/version"
                
                logger.info(f"Checking Ollama health at: {health_url}")
                health_check = requests.get(health_url, timeout=5)
                
                if health_check.status_code != 200:
                    logger.error(f"Ollama service returned non-200 status: {health_check.status_code}")
                    return Response({
                        "error": "Local language model service is unavailable. Please try again later."
                    }, status=status.HTTP_503_SERVICE_UNAVAILABLE)
                
                # Also check if the specific model is available
                model_check_url = f"{base_url}/api/tags"
                model_check = requests.get(model_check_url, timeout=5)
                
                if model_check.status_code == 200:
                    models = model_check.json().get('models', [])
                    model_names = [m.get('name', '') for m in models]
                    if 'llama3:8b' not in model_names:
                        logger.error(f"Required model 'llama3:8b' not found. Available models: {model_names}")
                        return Response({
                            "error": "Required model 'llama3:8b' is not available. Please run 'ollama pull llama3:8b' first."
                        }, status=status.HTTP_503_SERVICE_UNAVAILABLE)
                    
                logger.info("Ollama service and required model are available")
                
            except requests.exceptions.ConnectionError:
                logger.error("Connection error - Ollama service appears to be down or not running")
                return Response({
                    "error": "Cannot connect to local language model service. Please make sure Ollama is running with 'ollama serve'."
                }, status=status.HTTP_503_SERVICE_UNAVAILABLE)
            except Exception as e:
                logger.warning(f"Could not verify Ollama status: {str(e)}")
                # Continue anyway, we'll catch failures later
            
            logger.info(f"Looking up event with ID: {event_id}")
            event = Event.objects.get(id=event_id)
            
            # Check if the user has permission to access this event
            if not request.user.is_staff:  # Allow staff to access all events
                logger.warning(f"Permission denied for user {request.user.id} to access event {event_id}")
                return Response({"error": "Permission denied"}, status=status.HTTP_403_FORBIDDEN)
            
            # Check if worksheet_url is available
            if not event.worksheet_url:
                logger.error(f"Event {event_id} has no worksheet_url")
                return Response({"error": "Event has no associated worksheet URL"}, status=status.HTTP_400_BAD_REQUEST)
                
            # Extract data from the Google Sheet
            logger.info(f"Extracting data from worksheet: {event.worksheet_url}")
            df, data = self.extract_data_from_sheet(event.worksheet_url)
            
            if df is None or data is None:
                logger.error(f"Failed to extract data from Google Sheet for event {event_id}")
                return Response({
                    "error": "Could not extract data from Google Sheet. Make sure the sheet is accessible."
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Generate insights using improved RAG approach
            logger.info(f"Using RAG approach for dataset with {len(df)} records")
            insights = self.generate_insights_with_rag(df)
            
            if "error" in insights:
                logger.error(f"Error in insights generation: {insights.get('error')}")
                return Response({
                    "error": "Error generating insights",
                    "details": insights.get('error')
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
            logger.info(f"Successfully generated insights for event {event_id}")
            return Response({
                "event_name": event.name,
                "insights": insights,
                "processed_columns": list(df.columns) if isinstance(df, pd.DataFrame) else []
            })
            
        except Event.DoesNotExist:
            logger.error(f"Event not found with ID: {event_id}")
            return Response({"error": "Event not found"}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            logger.error(f"Unexpected error processing request: {str(e)}", exc_info=True)
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
