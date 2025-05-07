
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

# Configure logging
logger = logging.getLogger(__name__)

class GenerateInsightsView(APIView):
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]
    
    def extract_data_from_sheet(self, worksheet_url):
        """
        Extract data from Google Sheets
        For this function to work, the Google Sheet must be shared with "Anyone with the link"
        """
        try:
            logger.info(f"Attempting to extract data from: {worksheet_url}")
            
            # Convert Google Sheets URL to export format
            if "spreadsheets/d/" in worksheet_url:
                sheet_id = worksheet_url.split("spreadsheets/d/")[1].split("/")[0]
                export_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
                
                logger.info(f"Generated export URL: {export_url}")
                
                # Read data from CSV
                df = pd.read_csv(export_url)
                data = df.to_dict(orient='records')
                logger.info(f"Successfully extracted {len(data)} records from sheet")
                return df, data
            else:
                logger.error(f"Invalid Google Sheets URL format: {worksheet_url}")
                return None, None
        except Exception as e:
            logger.error(f"Error extracting data from sheet: {str(e)}", exc_info=True)
            return None, None
    
    def preprocess_feedback_data(self, df):
        """
        Preprocess feedback data by:
        1. Removing personally identifiable information (PII) columns
        2. Removing irrelevant columns
        3. Handling missing values
        """
        try:
            logger.info("Starting feedback data preprocessing")
            
            # Create a copy to avoid modifying original data
            processed_df = df.copy()
            
            # List of common PII/irrelevant columns to exclude (customize based on your data)
            common_exclude_columns = [
                'name', 'fullname', 'first name', 'last name', 'firstname', 'lastname', 
                'usn', 'registration', 'student id', 'studentid', 'id', 'email', 'phone', 
                'mobile', 'address', 'timestamp', 'submitted at', 'submissiondate',
                'ip address', 'age', 'roll no', 'dob', 'date of birth'
            ]
            
            # Case-insensitive column filtering
            columns_to_drop = []
            for col in processed_df.columns:
                if any(exclude.lower() in col.lower() for exclude in common_exclude_columns):
                    columns_to_drop.append(col)
            
            # Drop identified columns
            if columns_to_drop:
                logger.info(f"Dropping irrelevant columns: {columns_to_drop}")
                processed_df = processed_df.drop(columns=columns_to_drop, errors='ignore')
            
            # Handle missing values in text columns
            for col in processed_df.columns:
                if processed_df[col].dtype == 'object':  # If column is text/categorical
                    processed_df[col] = processed_df[col].fillna('No response')
            
            # For numeric columns, fill with median
            for col in processed_df.select_dtypes(include=[np.number]).columns:
                processed_df[col] = processed_df[col].fillna(processed_df[col].median())
                
            logger.info(f"Preprocessing complete. Retained {len(processed_df.columns)} relevant columns")
            return processed_df
        except Exception as e:
            logger.error(f"Error preprocessing feedback data: {str(e)}", exc_info=True)
            return df  # Return original data if preprocessing fails
    
    def generate_embeddings(self, texts):
        """
        Generate simple TF-IDF embeddings for text data
        """
        vectorizer = TfidfVectorizer(stop_words='english')
        return vectorizer.fit_transform(texts)
    
    def chunk_data(self, df, chunk_size=10):
        """
        Split the dataset into chunks for RAG processing
        """
        # Create chunks of the dataframe
        num_rows = len(df)
        chunks = []
        
        for i in range(0, num_rows, chunk_size):
            end_idx = min(i + chunk_size, num_rows)
            chunks.append(df.iloc[i:end_idx])
            
        logger.info(f"Split dataset into {len(chunks)} chunks")
        return chunks
        
    def extract_key_insights_from_chunk(self, chunk_df):
        """
        Extract key insights from a single chunk using Groq API
        """
        try:
            logger.info(f"Extracting insights from chunk with {len(chunk_df)} rows")
            
            api_key = os.environ.get('GROQ_API_KEY')
            if not api_key:
                logger.error("Groq API key not configured in environment variables")
                return {"error": "Groq API key not configured"}
            
            # Convert chunk to JSON string
            chunk_data = chunk_df.to_dict(orient='records')
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            # Create a focused prompt for this chunk
            prompt = f"""
            Analyze this chunk of feedback data:
            
            {json.dumps(chunk_data, indent=2)}
            
            Extract only the key insights and themes from this chunk. Focus only on:
            1. Main sentiment (positive/negative/neutral)
            2. Key issues or concerns mentioned
            3. Specific strengths highlighted
            4. Notable suggestions for improvement
            5. Any unique or standout feedback
            
            Be concise and focus only on extracting factual insights. Format as JSON.
            """
            
            payload = {
                "model": "llama3-70b-8192",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,  # Lower temperature for more factual analysis
                "max_tokens": 1000  # Reduced token usage for chunk analysis
            }
            
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                response_data = response.json()
                chunk_insights = response_data['choices'][0]['message']['content']
                
                # Try to parse as JSON
                try:
                    return json.loads(chunk_insights)
                except json.JSONDecodeError:
                    return {"raw_insights": chunk_insights}
            else:
                logger.error(f"API Error: {response.status_code}: {response.text}")
                return {"error": f"API Error: {response.status_code}"}
                
        except Exception as e:
            logger.error(f"Error extracting insights from chunk: {str(e)}", exc_info=True)
            return {"error": str(e)}
    
    def synthesize_insights(self, chunk_insights):
        """
        Synthesize all chunk insights into final comprehensive insights
        """
        try:
            logger.info("Synthesizing insights from all chunks")
            
            api_key = os.environ.get('GROQ_API_KEY')
            if not api_key:
                return {"error": "Groq API key not configured"}
                
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            # Prepare final synthesis prompt
            prompt = f"""
            Synthesize these insights from different chunks of feedback data:
            
            {json.dumps(chunk_insights, indent=2)}
            
            Create a comprehensive analysis with:
            1. Overall sentiment analysis (with approximate percentages if possible)
            2. Top 3-5 key areas needing improvement (with specific details)
            3. Top 3-5 strengths and positive aspects (with specific details)
            4. Clear patterns or trends across the feedback
            5. Actionable recommendations based on the feedback
            
            Format your response as a structured JSON with these sections.
            """
            
            payload = {
                "model": "llama3-70b-8192",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 2500
            }
            
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                response_data = response.json()
                final_insights = response_data['choices'][0]['message']['content']
                
                # Try to parse as JSON
                try:
                    return json.loads(final_insights)
                except json.JSONDecodeError:
                    return {"raw_insights": final_insights}
            else:
                return {"error": f"API Error: {response.status_code}"}
                
        except Exception as e:
            logger.error(f"Error synthesizing insights: {str(e)}", exc_info=True)
            return {"error": str(e)}
    
    def generate_insights_with_rag(self, df):
        """
        Generate insights using RAG approach to handle large datasets
        """
        try:
            # 1. Preprocess data to remove PII and irrelevant columns
            processed_df = self.preprocess_feedback_data(df)
            
            # 2. Split data into manageable chunks
            chunks = self.chunk_data(processed_df, chunk_size=15)  # Adjust chunk size as needed
            
            # 3. Process each chunk to extract insights
            chunk_insights = []
            for i, chunk in enumerate(chunks):
                logger.info(f"Processing chunk {i+1}/{len(chunks)}")
                insights = self.extract_key_insights_from_chunk(chunk)
                
                if "error" not in insights:
                    chunk_insights.append(insights)
                else:
                    logger.warning(f"Error processing chunk {i+1}: {insights.get('error')}")
            
            # 4. If no successful chunk insights, return error
            if not chunk_insights:
                return {"error": "Failed to extract insights from any data chunks"}
            
            # 5. Synthesize all chunk insights into comprehensive analysis
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
            
            # Check if dataset is too small for chunking
            if len(df) < 5:  # For very small datasets, use the original method
                logger.info("Dataset too small for RAG approach, using standard method")
                insights = self.generate_insights_with_groq(data)
            else:
                # Generate insights using RAG for larger datasets
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
    
    def generate_insights_with_groq(self, data):
        """
        Original method for small datasets (kept for backward compatibility)
        """
        try:
            logger.info("Beginning insights generation with Groq API")
            
            api_key = os.environ.get('GROQ_API_KEY')
            if not api_key:
                logger.error("Groq API key not configured in environment variables")
                return {"error": "Groq API key not configured"}
                
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            # Prepare prompt for the LLM
            prompt = f"""
            Analyze the following feedback data and provide actionable insights:
            
            {json.dumps(data, indent=2)}
            
            Please generate insights in the following categories:
            1. Overall sentiment analysis
            2. Key areas of improvement
            3. Strengths and positive feedback
            4. Trend analysis (if multiple responses over time)
            5. Recommended action items
            
            Format your response as structured JSON with sections for each category.
            """
            
            payload = {
                "model": "llama3-70b-8192",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 4000
            }
            
            logger.info("Sending request to Groq API")
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload
            )
            
            logger.info(f"Received response from Groq API with status code: {response.status_code}")
            
            if response.status_code == 200:
                response_data = response.json()
                insights = response_data['choices'][0]['message']['content']
                
                # Try to parse the response as JSON
                try:
                    insights_json = json.loads(insights)
                    logger.info("Successfully parsed insights as JSON")
                    return insights_json
                except json.JSONDecodeError as e:
                    logger.warning(f"Could not parse insights as JSON: {str(e)}")
                    # If can't parse as JSON, return as text
                    return {"raw_insights": insights}
            else:
                error_msg = f"API Error: {response.status_code}"
                logger.error(f"{error_msg}: {response.text}")
                return {"error": error_msg, "details": response.text}
                
        except Exception as e:
            logger.error(f"Error generating insights with Groq: {str(e)}", exc_info=True)
            return {"error": str(e)}