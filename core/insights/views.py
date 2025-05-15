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
import asyncio
import hashlib
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
        self.ollama_api_url = os.environ.get('OLLAMA_API_URL', 'http://localhost:11434/api/generate')
        self.vectorizer = TfidfVectorizer(stop_words='english')
        # Adjust these values based on logs
        self.chunk_size = int(os.environ.get('INSIGHT_CHUNK_SIZE', '5'))  # Reduced chunk size
        self.max_workers = min(int(os.environ.get('MAX_OLLAMA_WORKERS', '2')), 3)  # Reduced workers
        self.max_tokens_chunk = int(os.environ.get('MAX_TOKENS_CHUNK', '600'))  # Reduced tokens
        self.max_tokens_synthesis = int(os.environ.get('MAX_TOKENS_SYNTHESIS', '1000'))  # Reduced tokens
        self.request_timeout = int(os.environ.get('OLLAMA_TIMEOUT', '30'))  # Reduced timeout
        
        # Add a small cache for request results
        self.cache = {}
        
        # Log configuration
        logger.info(f"Initialized with: Ollama API URL={self.ollama_api_url}, "
                   f"Max workers={self.max_workers}, Chunk size={self.chunk_size}, "
                   f"Request timeout={self.request_timeout}s")
    def calculate_prompt_hash(self, prompt, max_tokens, temperature):
        """Generate a consistent hash for prompt caching"""
        hash_input = f"{prompt}|{max_tokens}|{temperature}"
        return hashlib.md5(hash_input.encode()).hexdigest()
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
        Make API request with improved retry logic and better error handling
        """
        # Generate hash for caching
        prompt_hash = self.calculate_prompt_hash(prompt, max_tokens, temperature)
        
        # Check cache first
        if prompt_hash in self.cache:
            logger.info(f"Using cached result for prompt hash: {prompt_hash[:8]}...")
            return self.cache[prompt_hash]
        
        # Implement progressive timeouts
        base_timeout = min(self.request_timeout, 30)  # Cap initial timeout
        
        for attempt in range(retries + 1):
            try:
                headers = {"Content-Type": "application/json"}
                
                # Use a smaller prompt and token limit for retries
                adjusted_prompt = prompt
                adjusted_max_tokens = max_tokens
                
                if attempt > 0:
                    # For retries: reduce prompt complexity and token limit
                    prompt_parts = prompt.split("\n\n")
                    if len(prompt_parts) > 2:
                        # Take first two parts and the data part
                        adjusted_prompt = "\n\n".join(prompt_parts[:2] + [prompt_parts[-1]])
                    adjusted_max_tokens = max(300, max_tokens // 2)  # Reduce token limit for faster response
                
                payload = {
                    "model": "llama3:8b",
                    "prompt": adjusted_prompt,
                    "stream": False,
                    "temperature": temperature,
                    "max_tokens": adjusted_max_tokens
                }
                
                # Calculate timeout with progressive backoff
                current_timeout = base_timeout * (1 + attempt * 0.5)
                logger.info(f"Sending request to Ollama API (attempt {attempt+1}, timeout: {current_timeout:.1f}s)")
                
                response = requests.post(
                    self.ollama_api_url,
                    headers=headers,
                    json=payload,
                    timeout=current_timeout
                )
                
                if response.status_code == 200:
                    response_data = response.json()
                    result = response_data.get('response', '')
                    
                    # Cache successful result
                    self.cache[prompt_hash] = result
                    
                    # Log some stats about the response
                    result_length = len(result)
                    logger.info(f"Received response ({result_length} chars) from Ollama API")
                    
                    return result
                else:
                    error_text = response.text[:200]
                    logger.warning(f"API request failed (attempt {attempt+1}): HTTP {response.status_code} - {error_text}")
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Request timeout (attempt {attempt+1}) after {base_timeout * (1 + attempt * 0.5):.1f}s")
            except requests.exceptions.ConnectionError:
                logger.warning(f"Connection error (attempt {attempt+1}) - Ollama service may be overloaded")
            except Exception as e:
                logger.warning(f"Request error (attempt {attempt+1}): {str(e)}")
                
            # Only sleep if we're going to retry
            if attempt < retries:
                import time
                time.sleep(2 * (attempt + 1))  # Exponential backoff
        
        # All retries failed, return simpler fallback analysis
        logger.warning("All API request attempts failed, returning fallback analysis")
        return self._generate_fallback_analysis(prompt)
    def _generate_fallback_analysis(self, prompt):
        """Generate simple fallback analysis when Ollama API fails"""
        # Extract any data from the prompt
        try:
            # Look for JSON data in the prompt
            import re
            data_match = re.search(r'{[\s\S]*}', prompt)
            if data_match:
                data_str = data_match.group(0)
                data = json.loads(data_str)
                
                # Simple text analysis
                if isinstance(data, list) and len(data) > 0:
                    # Count feedback items
                    count = len(data)
                    
                    # Generate simple fallback response
                    return json.dumps({
                        "sentiment": {
                            "note": "Auto-generated due to API timeout",
                            "summary": "Mixed feedback detected"
                        },
                        "key_issues": [
                            "Unable to perform detailed analysis due to processing timeout",
                            "System recommends reviewing the raw feedback directly"
                        ],
                        "strengths": [
                            "Basic sentiment analysis indicates mixed feedback"
                        ],
                        "patterns": [
                            f"Dataset contains {count} feedback items that require manual review"
                        ]
                    })
        except:
            pass
        
        # If we can't parse the data, return minimal response
        return json.dumps({
            "error": "Analysis timed out",
            "recommendation": "Please try reducing the dataset size or review feedback manually"
        })
    
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

        json_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
        json_matches = re.findall(json_pattern, text)
        
        if json_matches:
            for json_text in json_matches:
                try:
                    # Try to parse each match
                    json_obj = json.loads(json_text.strip())
                    return json_text.strip()
                except json.JSONDecodeError:
                    continue
        
        # Method 2: Try to find outermost JSON object
        try:
            # Find the first opening brace and corresponding closing brace
            start = text.find("{")
            if start >= 0:
                # Find the matching closing brace
                stack = 1
                for i in range(start + 1, len(text)):
                    if text[i] == "{":
                        stack += 1
                    elif text[i] == "}":
                        stack -= 1
                        if stack == 0:
                            # Found matching brace
                            potential_json = text[start:i+1]
                            try:
                                json.loads(potential_json)
                                return potential_json
                            except json.JSONDecodeError:
                                pass
        except:
            pass
        
        # Method 3: Last resort - try to fix common JSON issues
        try:
            # Replace common formatting issues
            cleaned_text = text
            
            # Fix missing quotes around keys
            key_pattern = r'(\s*?)(\w+)(\s*?):'
            cleaned_text = re.sub(key_pattern, r'\1"\2"\3:', cleaned_text)
            
            # Fix single quotes being used instead of double quotes
            cleaned_text = cleaned_text.replace("'", '"')
            
            # Extract with { } pattern
            if "{" in cleaned_text and "}" in cleaned_text:
                start = cleaned_text.find("{")
                end = cleaned_text.rfind("}") + 1
                if start < end:
                    try:
                        json_obj = json.loads(cleaned_text[start:end])
                        return cleaned_text[start:end]
                    except:
                        pass
        except:
            pass
            
        # If all else fails, create a simple JSON structure with the text
        return json.dumps({"raw_text": text[:2000]})  # Truncate if too long
    
    def process_chunks_parallel(self, chunks, feedback_cols=None):
        """
        Process chunks in parallel with better error handling and partial results
        """
        chunk_insights = []
        failed_chunks = 0
        
        # Use dynamic worker pool based on available chunks
        num_workers = min(self.max_workers, max(1, len(chunks) // 2))
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Process small batches with shorter timeouts first
            futures = []
            for i, chunk in enumerate(chunks):
                futures.append(executor.submit(self.extract_key_insights_from_chunk, chunk, feedback_cols))
            
            # Process results as they complete
            for i, future in enumerate(as_completed(futures)):
                try:
                    insight = future.result()
                    if "error" not in insight:
                        logger.info(f"Successfully processed chunk {i+1}/{len(chunks)}")
                        chunk_insights.append(insight)
                    else:
                        logger.warning(f"Failed to process chunk {i+1}: {insight.get('error')}")
                        
                        # Try a simpler approach for failed chunks as fallback
                        if len(chunks[i]) <= 5:  # Only attempt fallback for small chunks
                            simplified_insight = self._process_chunk_fallback(chunks[i])
                            if "error" not in simplified_insight:
                                logger.info(f"Fallback processing succeeded for chunk {i+1}")
                                chunk_insights.append(simplified_insight)
                            else:
                                failed_chunks += 1
                        else:
                            failed_chunks += 1
                except Exception as e:
                    logger.error(f"Exception processing chunk {i+1}: {str(e)}")
                    failed_chunks += 1
        
        logger.info(f"Completed chunk processing: {len(chunk_insights)} successful, {failed_chunks} failed")
        return chunk_insights
    def _process_chunk_fallback(self, chunk_df):
        """Simple fallback processing for failed chunks"""
        try:
            # Create a much simpler analysis based on basic statistics
            text_cols = chunk_df.select_dtypes(include=['object']).columns
            
            # Count non-empty responses
            response_counts = {}
            for col in text_cols:
                non_empty = chunk_df[col].str.len() > 10
                response_counts[col] = non_empty.sum()
            
            # Find most common words (simple frequency analysis)
            common_words = {}
            for col in text_cols:
                text = " ".join(chunk_df[col].fillna("").astype(str).tolist())
                words = re.findall(r'\b\w+\b', text.lower())
                word_freq = {}
                for word in words:
                    if word not in ["the", "and", "to", "of", "is", "in", "for", "a", "with"]:
                        word_freq[word] = word_freq.get(word, 0) + 1
                
                # Get top 5 words
                top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
                common_words[col] = [w[0] for w in top_words]
            
            return {
                "sentiment": "neutral",
                "fallback_analysis": True,
                "response_counts": response_counts,
                "common_words": common_words
            }
            
        except Exception as e:
            logger.error(f"Error in chunk fallback processing: {str(e)}")
            return {"error": "Both main and fallback processing failed"}
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
            # Check if model is available
            try:
                # Extract base URL (remove endpoint path)
                base_url = self.ollama_api_url.split('/api/')[0]
                health_url = f"{base_url}/api/version"
                
                logger.info(f"Checking Ollama health at: {health_url}")
                health_check = requests.get(health_url, timeout=5)
                
                if health_check.status_code != 200:
                    logger.error(f"Ollama service returned non-200 status: {health_check.status_code}")
                    return Response({
                        "error": "Local language model service is unavailable",
                        "details": "Please ensure the Ollama service is running correctly"
                    }, status=status.HTTP_503_SERVICE_UNAVAILABLE)
                    
            except requests.exceptions.RequestException:
                logger.error("Connection error - Ollama service appears to be down")
                return Response({
                    "error": "Cannot connect to local language model service",
                    "details": "Please make sure Ollama is running with 'ollama serve'"
                }, status=status.HTTP_503_SERVICE_UNAVAILABLE)
                
            # Continue with request processing
            logger.info(f"Looking up event with ID: {event_id}")
            event = Event.objects.get(id=event_id)
            
            # Permission check
            if not request.user.is_staff:
                logger.warning(f"Permission denied for user {request.user.id} to access event {event_id}")
                return Response({"error": "Permission denied"}, status=status.HTTP_403_FORBIDDEN)
            
            # Check for worksheet URL
            if not event.worksheet_url:
                logger.error(f"Event {event_id} has no worksheet_url")
                return Response({"error": "Event has no associated worksheet URL"}, 
                            status=status.HTTP_400_BAD_REQUEST)
                
            # Extract data with timeout protection
            try:
                df, data = self.extract_data_from_sheet(event.worksheet_url)
                
                if df is None or data is None:
                    logger.error(f"Failed to extract data from Google Sheet for event {event_id}")
                    return Response({
                        "error": "Could not extract data from Google Sheet",
                        "details": "Please ensure the sheet is publicly accessible or shared properly"
                    }, status=status.HTTP_400_BAD_REQUEST)
                
                # Handle very large datasets with a warning and sample
                if len(df) > 100:
                    logger.warning(f"Large dataset detected ({len(df)} rows). Sampling to 100 rows.")
                    df = df.sample(n=100, random_state=42) if len(df) > 100 else df
                    
            except Exception as e:
                logger.error(f"Error extracting sheet data: {str(e)}")
                return Response({
                    "error": "Failed to process Google Sheet data",
                    "details": str(e)
                }, status=status.HTTP_400_BAD_REQUEST)
                
            # Process with overall timeout protection
            try:
                # Generate insights with timeout protection
                import threading
                import queue
                
                result_queue = queue.Queue()
                
                def process_with_timeout():
                    try:
                        insights = self.generate_insights_with_rag(df)
                        result_queue.put(("success", insights))
                    except Exception as e:
                        logger.error(f"Error in RAG processing: {str(e)}")
                        result_queue.put(("error", str(e)))
                
                # Start processing in a separate thread
                process_thread = threading.Thread(target=process_with_timeout)
                process_thread.daemon = True
                process_thread.start()
                
                # Wait for results with a timeout (3 minutes)
                overall_timeout = 180
                process_thread.join(overall_timeout)
                
                if process_thread.is_alive():
                    logger.error(f"Processing timed out after {overall_timeout} seconds")
                    
                    # Generate fallback insights with basic stats
                    fallback_insights = self._generate_basic_stats(df)
                    
                    return Response({
                        "event_name": event.name,
                        "insights": fallback_insights,
                        "warning": "Analysis timed out. Showing simplified results.",
                        "processed_columns": list(df.columns)
                    })
                    
                # Get results from queue
                if not result_queue.empty():
                    status_code, insights = result_queue.get()
                    
                    if status_code == "error":
                        logger.error(f"Error in insights generation: {insights}")
                        return Response({
                            "error": "Error generating insights",
                            "details": insights
                        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
                    
                    if "error" in insights:
                        logger.error(f"Error in insights object: {insights.get('error')}")
                        
                        # Try to provide partial results if possible
                        if "raw_insights" in insights:
                            return Response({
                                "event_name": event.name,
                                "partial_insights": True,
                                "insights": {"raw_analysis": insights.get("raw_insights")},
                                "processed_columns": list(df.columns)
                            })
                        
                        return Response({
                            "error": "Error generating insights",
                            "details": insights.get('error')
                        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
                    
                    logger.info(f"Successfully generated insights for event {event_id}")
                    return Response({
                        "event_name": event.name,
                        "insights": insights,
                        "processed_columns": list(df.columns)
                    })
                else:
                    # Should never happen
                    logger.error("Process thread completed but no results in queue")
                    return Response({
                        "error": "No results generated",
                        "details": "Processing completed but no data returned"
                    }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
                    
            except Exception as e:
                logger.error(f"Unexpected error in insights processing: {str(e)}", exc_info=True)
                return Response({
                    "error": "Error processing insights",
                    "details": str(e)
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
                
        except Event.DoesNotExist:
            logger.error(f"Event not found with ID: {event_id}")
            return Response({"error": "Event not found"}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            logger.error(f"Unexpected error processing request: {str(e)}", exc_info=True)
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
    def _generate_basic_stats(self, df):
        """Generate very basic insights when full processing fails"""
        try:
            # Count responses
            response_count = len(df)
            
            # Find text columns (potential feedback columns)
            text_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            # Calculate response rates and lengths
            stats = {}
            for col in text_cols:
                # Count non-empty responses
                non_empty = df[col].notna() & (df[col].astype(str).str.len() > 3)
                response_rate = (non_empty.sum() / len(df)) * 100
                
                # Calculate average response length
                avg_length = df.loc[non_empty, col].astype(str).str.len().mean()
                
                stats[col] = {
                    "response_rate": f"{response_rate:.1f}%",
                    "avg_length": f"{avg_length:.1f} chars" if not pd.isna(avg_length) else "N/A"
                }
            
            return {
                "basic_stats": {
                    "total_responses": response_count,
                    "column_stats": stats
                },
                "note": "Full analysis timed out. Showing basic statistics only."
            }
        except Exception as e:
            logger.error(f"Error generating basic stats: {str(e)}")
            return {
                "error": "Unable to generate even basic statistics",
                "message": "Please try again with a smaller dataset"
            }