# views.py
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework.authentication import TokenAuthentication
from home.models import *
import pandas as pd
import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables

class GenerateInsightsView(APIView):
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]
    
    def extract_data_from_sheet(self, worksheet_url):
        """
        Extract data from Google Sheets
        For this function to work, the Google Sheet must be shared with "Anyone with the link"
        """
        try:
            # Convert Google Sheets URL to export format
            if "spreadsheets/d/" in worksheet_url:
                sheet_id = worksheet_url.split("spreadsheets/d/")[1].split("/")[0]
                export_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
                
                # Read data from CSV
                df = pd.read_csv(export_url)
                return df.to_dict(orient='records')
            else:
                return None
        except Exception as e:
            print(f"Error extracting data: {str(e)}")
            return None
    
    def generate_insights_with_groq(self, data):
        """
        Generate insights from data using Groq API
        """
        try:
            api_key = os.environ.get('GROQ_API_KEY')
            if not api_key:
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
                "model": "llama3-70b-8192",  # You can choose an appropriate model
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 4000
            }
            
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                data=json.dumps(payload)
            )
            
            if response.status_code == 200:
                response_data = response.json()
                insights = response_data['choices'][0]['message']['content']
                
                # Try to parse the response as JSON
                try:
                    insights_json = json.loads(insights)
                    return insights_json
                except:
                    # If can't parse as JSON, return as text
                    return {"raw_insights": insights}
            else:
                return {"error": f"API Error: {response.status_code}", "details": response.text}
                
        except Exception as e:
            return {"error": str(e)}
    
    def post(self, request, *args, **kwargs):
        event_id = request.data.get('event_id')
        
        if not event_id:
            return Response({"error": "Event ID is required"}, status=400)
        
        try:
            event = Event.objects.get(id=event_id)
            
            # Check if the user has permission to access this event
            if not request.user.is_staff:  # Allow staff to access all events
                return Response({"error": "Permission denied"}, status=403)
            
            # Extract data from the Google Sheet
            data = self.extract_data_from_sheet(event.worksheet_url)
            
            if not data:
                return Response({
                    "error": "Could not extract data from Google Sheet. Make sure the sheet is accessible."
                }, status=400)
            
            # Generate insights using Groq
            insights = self.generate_insights_with_groq(data)
            
            return Response({
                "event_name": event.name,
                "insights": insights
            })
            
        except Event.DoesNotExist:
            return Response({"error": "Event not found"}, status=404)
        except Exception as e:
            return Response({"error": str(e)}, status=500)
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)