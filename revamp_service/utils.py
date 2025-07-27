import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any
import pandas as pd
from revamp_service.prompts import *
import io
from revamp_service.configs import *
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from email.mime.base import MIMEBase
from email import encoders
import pandas as pd
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import requests
from io import StringIO
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from revamp_service.models import *
from revamp_service.logger import *
from revamp_service.analyzer import OllamaRAGAnalyzer
from revamp_service.baseAnalyzer import *
from .configs import *
logging = get_logger(__name__)

def send_error_email(recipient_email: str, error_msg: str, event_name: str):
    config = Mailconfig()
    try:
        subject = f"❌ Feedback Analysis Failed - {event_name}"
        body = f"""
        <html>
        <body>
            <h2>Analysis Failed</h2>
            <p>The feedback analysis for {event_name} encountered an error:</p>
            <p><strong>Error:</strong> {error_msg}</p>
            <p>Please contact support for assistance.</p>
        </body>
        </html>
        """
        
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = config.FROM_EMAIL
        msg['To'] = recipient_email
        
        html_part = MIMEText(body, 'html')
        msg.attach(html_part)
        
        loop = asyncio.get_event_loop()
        
        def _send_email():
            with smtplib.SMTP(config.SMTP_SERVER, config.SMTP_PORT) as server:
                server.starttls()
                server.login(config.EMAIL_USER, config.EMAIL_PASSWORD)
                server.send_message(msg)
        
            loop.run_in_executor(None, _send_email)
        
    except Exception as e:
        logging.error(f"Failed to send error email: {str(e)}")


def dataframe_to_text_rows(df):
    headers = list(df.columns)
    rows = []
    for _, row in df.iterrows():
        row_text = " | ".join([f"{header}: {value}" for header, value in zip(headers, row)])
        rows.append(row_text)
    return rows

async def fetch_worksheet_data(worksheet_url: str) -> pd.DataFrame:
    """Async version of fetch_worksheet_data"""
    loop = asyncio.get_event_loop()
    
    def _fetch():
        try:
            if 'docs.google.com/spreadsheets' in worksheet_url:
                sheet_id = worksheet_url.split('/d/')[1].split('/')[0]
                csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
            else:
                csv_url = worksheet_url
            logging.debug(f" Fetching data from: {csv_url}")
            print(f"📥 Fetching data from: {csv_url}")
            response = requests.get(csv_url, timeout=30)
            response.raise_for_status()
            
            df = pd.read_csv(StringIO(response.text))
            logging.debug(f"✅ Successfully loaded {len(df)} rows and {len(df.columns)} columns")
            print(f"✅ Successfully loaded {len(df)} rows and {len(df.columns)} columns")
            return df
            
        except Exception as e:
            logging.error(f"Error fetching worksheet data: {str(e)}")
            raise Exception(f"Failed to fetch worksheet data: {str(e)}")
    
    return await loop.run_in_executor(None, _fetch)
def generate_pdf_report(results: Dict[str, Any], event_name: str) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#667eea'),
        alignment=TA_CENTER,
        spaceAfter=30
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#764ba2'),
        spaceBefore=20,
        spaceAfter=10
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=14,
        textColor=colors.HexColor('#667eea'),
        spaceBefore=15,
        spaceAfter=8
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=11,
        textColor=colors.black,
        alignment=TA_LEFT,
        spaceAfter=8
    )
    
    story = []
    
    story.append(Paragraph(f"📊 FEEDBACK ANALYSIS REPORT", title_style))
    story.append(Paragraph(f"Event: {event_name}", heading_style))
    story.append(Spacer(1, 20))
    
    type_counts = {}
    for col, data in results.items():
        col_type = data.get('type', 'unknown')
        type_counts[col_type] = type_counts.get(col_type, 0) + 1
    
    story.append(Paragraph("📋 OVERVIEW", heading_style))
    story.append(Paragraph(f"Total Columns Analyzed: {len(results)}", body_style))
    
    for col_type, count in type_counts.items():
        story.append(Paragraph(f"{col_type.title()} Columns: {count}", body_style))
    
    story.append(Spacer(1, 20))
    
    for column, data in results.items():
        if 'error' in data:
            continue
            
        story.append(Paragraph(f"{column.upper()}", heading_style))
        story.append(Paragraph(f"Type: {data['type'].title()}", subheading_style))
        
        if data['type'] in ['numerical', 'rating']:
            analysis = data['analysis']
            stats_data = [
                ['Total Responses', str(analysis['total_responses'])],
                ['Average Score', str(analysis['mean'])],
                ['Median', str(analysis['median'])],
                ['Standard Deviation', str(analysis['std_dev'])]
            ]
            
            if 'rating_distribution' in analysis and analysis.get('mode'):
                stats_data.append(['Most Common Rating', str(analysis.get('mode', 'N/A'))])
            
            stats_table = Table(stats_data, colWidths=[2.5*inch, 1.5*inch])
            stats_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f0f4ff')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#667eea')),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e1e8f0'))
            ]))
            story.append(stats_table)
        
        elif data['type'] == 'categorical':
            analysis = data['analysis']
            stats_data = [
                ['Total Responses', str(analysis['total_responses'])],
                ['Most Selected', str(analysis['most_common'])],
                ['Total Options', str(analysis['unique_categories'])]
            ]
            
            stats_table = Table(stats_data, colWidths=[2.5*inch, 1.5*inch])
            stats_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f0f4ff')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#667eea')),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e1e8f0'))
            ]))
            story.append(stats_table)
        
        elif data['type'] == 'text':
            analysis = data['analysis']
            stats_data = [
                ['Total Responses', str(analysis['total_responses'])],
                ['Average Length', f"{analysis['avg_length']:.0f} characters"],
                ['Average Word Count', f"{analysis['avg_word_count']:.0f} words"]
            ]
            
            stats_table = Table(stats_data, colWidths=[2.5*inch, 1.5*inch])
            stats_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f0f4ff')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#667eea')),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e1e8f0'))
            ]))
            story.append(stats_table)
        
        story.append(Spacer(1, 12))
        story.append(Paragraph("🔍 Key Insights", subheading_style))
        
        insights_text = data['insights'].replace('•', '• ').replace('\n', '<br/>')
        story.append(Paragraph(insights_text, body_style))
        story.append(Spacer(1, 20))
    
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

def generate_summary_report(results: Dict[str, Any]) -> str:
    html_parts = []
    
    html_parts.append("""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #333; max-width: 1200px; margin: 0 auto; padding: 20px; background-color: #f5f7fa; }
            .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 40px; border-radius: 15px; text-align: center; margin-bottom: 30px; box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3); }
            .header h1 { margin: 0; font-size: 2.5em; font-weight: 700; }
            .header p { margin: 10px 0 0 0; font-size: 1.2em; opacity: 0.9; }
            .overview { background: linear-gradient(135deg, #f8f9ff 0%, #e8f2ff 100%); padding: 25px; border-radius: 12px; margin-bottom: 30px; border: 1px solid #e1e8f0; }
            .overview h2 { color: #667eea; margin-top: 0; font-size: 1.8em; font-weight: 600; }
            .overview p { margin: 8px 0; font-size: 1.1em; }
            .column-section { background: white; margin-bottom: 25px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.08); overflow: hidden; border: 1px solid #e8eef5; }
            .column-header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; margin: 0; font-size: 1.4em; font-weight: 600; }
            .column-content { padding: 25px; }
            .stats { display: flex; gap: 20px; margin: 20px 0; flex-wrap: wrap; }
            .stat-item { background: linear-gradient(135deg, #f0f4ff 0%, #e8f2ff 100%); padding: 15px 20px; border-radius: 10px; flex: 1; min-width: 140px; border: 1px solid #d6e3f0; }
            .stat-label { font-size: 13px; color: #5a6c7d; text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px; }
            .stat-value { font-size: 24px; font-weight: 700; color: #667eea; margin-top: 5px; }
            .insights { background: #f9fafb; padding: 20px; border-radius: 10px; margin-top: 20px; border-left: 4px solid #667eea; }
            .insights h4 { color: #667eea; margin-top: 0; font-size: 1.2em; font-weight: 600; }
            .insights p { margin: 10px 0 0 0; font-size: 1em; line-height: 1.7; }
            .insights ul { margin: 10px 0; padding-left: 20px; }
            .insights li { margin: 8px 0; font-size: 1em; line-height: 1.6; }
            .type-badge { background: linear-gradient(135deg, #764ba2 0%, #667eea 100%); color: white; padding: 6px 16px; border-radius: 25px; font-size: 12px; display: inline-block; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
            .highlight { font-weight: 600; color: #667eea; }
            .metric-highlight { font-weight: 700; color: #2c3e50; }
        </style>
    </head>
    <body>
    """)
    
    html_parts.append("""
        <div class="header">
            <h1>📊 FEEDBACK ANALYSIS REPORT</h1>
            <p>Comprehensive analysis of participant feedback and responses</p>
        </div>
    """)
    
    type_counts = {}
    for col, data in results.items():
        col_type = data.get('type', 'unknown')
        type_counts[col_type] = type_counts.get(col_type, 0) + 1
    
    html_parts.append(f"""
        <div class="overview">
            <h2>📋 OVERVIEW</h2>
            <p><span class="highlight">Total Columns Analyzed:</span> <span class="metric-highlight">{len(results)}</span></p>
    """)
    
    for col_type, count in type_counts.items():
        html_parts.append(f"<p><span class=\"highlight\">{col_type.title()} Columns:</span> <span class=\"metric-highlight\">{count}</span></p>")
    
    html_parts.append("</div>")
    
    for column, data in results.items():
        if 'error' in data:
            continue
            
        html_parts.append(f"""
            <div class="column-section">
                <h2 class="column-header">{column.upper()}</h2>
                <div class="column-content">
                    <span class="type-badge">{data['type'].title()}</span>
                    <div class="stats">
        """)
        
        if data['type'] in ['numerical', 'rating']:
            analysis = data['analysis']
            html_parts.append(f"""
                        <div class="stat-item">
                            <div class="stat-label">Total Responses</div>
                            <div class="stat-value">{analysis['total_responses']}</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-label">Average Score</div>
                            <div class="stat-value">{analysis['mean']}</div>
                        </div>
            """)
            if 'rating_distribution' in analysis and analysis.get('mode'):
                html_parts.append(f"""
                        <div class="stat-item">
                            <div class="stat-label">Most Common Rating</div>
                            <div class="stat-value">{analysis.get('mode', 'N/A')}</div>
                        </div>
                """)
        
        elif data['type'] == 'categorical':
            analysis = data['analysis']
            html_parts.append(f"""
                        <div class="stat-item">
                            <div class="stat-label">Total Responses</div>
                            <div class="stat-value">{analysis['total_responses']}</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-label">Most Selected</div>
                            <div class="stat-value">{analysis['most_common']}</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-label">Total Options</div>
                            <div class="stat-value">{analysis['unique_categories']}</div>
                        </div>
            """)
        
        elif data['type'] == 'text':
            analysis = data['analysis']
            html_parts.append(f"""
                        <div class="stat-item">
                            <div class="stat-label">Total Responses</div>
                            <div class="stat-value">{analysis['total_responses']}</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-label">Average Length</div>
                            <div class="stat-value">{analysis['avg_length']:.0f}</div>
                        </div>
            """)
        
        formatted_insights = data['insights'].replace('\n', '<br>')
        formatted_insights = formatted_insights.replace('•', '<li>').replace('<li>', '</li><li>').replace('</li><li>', '<li>', 1)
        if formatted_insights.startswith('<li>'):
            formatted_insights = '<ul>' + formatted_insights + '</ul>'
        
        html_parts.append(f"""
                    </div>
                    <div class="insights">
                        <h4>🔍 Key Insights</h4>
                        <div>{formatted_insights}</div>
                    </div>
                </div>
            </div>
        """)
    
    html_parts.append("""
        </body>
        </html>
    """)
    
    return "".join(html_parts)

async def send_analysis_email(recipient_email: str, report: str, event_name: str, results: Dict[str, Any]):
    config = Mailconfig()
    try:
        if not config.EMAIL_USER or not config.EMAIL_PASSWORD:
            raise Exception("EMAIL_USER and EMAIL_PASSWORD must be set in environment variables")
        
        msg = MIMEMultipart('alternative')
        msg['Subject'] = f"📊 Feedback Analysis Report - {event_name}"
        msg['From'] = config.FROM_EMAIL
        msg['To'] = recipient_email
        
        html_part = MIMEText(report, 'html')
        msg.attach(html_part)
        
        pdf_data = generate_pdf_report(results, event_name)
        
        pdf_attachment = MIMEBase('application', 'pdf')
        pdf_attachment.set_payload(pdf_data)
        encoders.encode_base64(pdf_attachment)
        pdf_attachment.add_header(
            'Content-Disposition',
            f'attachment; filename="feedback_analysis_{event_name.replace(" ", "_")}.pdf"'
        )
        msg.attach(pdf_attachment)
        
        loop = asyncio.get_event_loop()
        
        def _send_email():
            try:
                print(f"🔄 Connecting to {config.SMTP_SERVER}:{config.SMTP_PORT}")
                
                with smtplib.SMTP(config.SMTP_SERVER, config.SMTP_PORT) as server:
                    server.timeout = 30
                    
                    print("🔐 Starting TLS...")
                    server.starttls()
                    
                    print(f"🔑 Logging in as: {config.EMAIL_USER}")
                    server.login(str(config.EMAIL_USER), str(config.EMAIL_PASSWORD))
                    
                    print(f"📤 Sending to: {recipient_email}")
                    server.send_message(msg)
                    
                    print("✅ Email sent successfully!")
                    
            except smtplib.SMTPAuthenticationError as auth_error:
                print(f"❌ Authentication failed: {auth_error}")
                print("💡 If using Gmail, make sure you're using an App Password!")
                raise Exception(f"SMTP Authentication failed: {auth_error}")
                
            except smtplib.SMTPException as smtp_error:
                print(f"❌ SMTP error: {smtp_error}")
                raise Exception(f"SMTP error: {smtp_error}")
                
            except Exception as general_error:
                print(f"❌ General error: {general_error}")
                raise Exception(f"Email sending failed: {general_error}")
        
        await loop.run_in_executor(None, _send_email)
        print(f"✅ Analysis report sent to {recipient_email}")
        return True
        
    except Exception as e:
        logging.error(f"Error sending email: {str(e)}")
        print(f"❌ Failed to send email: {str(e)}")
        return False
async def send_no_row_email(
    recipient_email: str,
    event_name: str,
    config: Config,
    row_count: int
):
    try:
        if not config.EMAIL_USER or not config.EMAIL_PASSWORD:
            raise Exception("EMAIL_USER and EMAIL_PASSWORD must be set in environment variables")

        msg = MIMEMultipart('alternative')
        msg['Subject'] = f"📊 Feedback Analysis Report - {event_name}"
        msg['From'] = config.FROM_EMAIL
        msg['To'] = recipient_email

        # Add warning if dataset is empty
        if row_count == 0:
            warning_html = """
                <div style="padding:10px; background-color:#ffe5e5; border-left:6px solid #ff0000; margin-bottom:10px;">
                    ⚠️ <strong>Notice:</strong> The provided dataset has <strong>0 rows</strong>. 
                    The analysis results may be empty or incomplete.
                </div>
            """
            report = warning_html + report

        html_part = MIMEText(report, 'html')
        msg.attach(html_part)

        loop = asyncio.get_event_loop()

        def _send_email():
            try:
                print(f"🔄 Connecting to {config.SMTP_SERVER}:{config.SMTP_PORT}")
                with smtplib.SMTP(config.SMTP_SERVER, config.SMTP_PORT) as server:
                    server.timeout = 30
                    print("🔐 Starting TLS...")
                    server.starttls()
                    print(f"🔑 Logging in as: {config.EMAIL_USER}")
                    server.login(str(config.EMAIL_USER), str(config.EMAIL_PASSWORD))
                    print(f"📤 Sending to: {recipient_email}")
                    server.send_message(msg)
                    print("✅ Email sent successfully!")
            except smtplib.SMTPAuthenticationError as auth_error:
                print(f"❌ Authentication failed: {auth_error}")
                raise Exception(f"SMTP Authentication failed: {auth_error}")
            except smtplib.SMTPException as smtp_error:
                print(f"❌ SMTP error: {smtp_error}")
                raise Exception(f"SMTP error: {smtp_error}")
            except Exception as general_error:
                print(f"❌ General error: {general_error}")
                raise Exception(f"Email sending failed: {general_error}")

        await loop.run_in_executor(None, _send_email)
        print(f"✅ Analysis report sent to {recipient_email}")
        return True

    except Exception as e:
        logging.error(f"Error sending email: {str(e)}")
        print(f"❌ Failed to send email: {str(e)}")
        return False

async def process_analysis_task(request: AnalysisRequest, task_id: str):
    try:
        print_terminal_separator(f"🎯 RAG FEEDBACK ANALYSIS STARTED - Task: {task_id}")
        logging.debug(f"Starting analysis task {task_id}")
        
        # Add timeout to prevent tasks from running indefinitely
        async with asyncio.timeout(1800):  # 30 minute timeout
            analyzer : Analyzer = OllamaRAGAnalyzer()
            
            df = await fetch_worksheet_data(request.worksheet_url)
            if len(df) > analyzer.config.MAX_PROCESSING_ROWS:
                print(f"⚠️ Limiting analysis to {analyzer.config.MAX_PROCESSING_ROWS} rows")
                df = df.head(analyzer.config.MAX_PROCESSING_ROWS)
            
            processed_df, column_types = analyzer.preprocess_columns(df)
            
            if processed_df.empty:
                logging.error("No relevant Columns")
                await send_no_row_email(
                request.recipient_email, 
                request.event_name,
                results
            )
            
            results = await analyze_columns_parallel(analyzer, processed_df, column_types)
            
            summary_report = generate_summary_report(results)
            
            await send_analysis_email(
                request.recipient_email, 
                summary_report, 
                request.event_name,
                results
            )
            
            print_terminal_separator(f"✅ ANALYSIS COMPLETE - Task: {task_id}")
            logging.info(f"Task {task_id} completed successfully")
            
    except asyncio.TimeoutError:
        logging.error(f"Task {task_id} timed out after 30 minutes")
        await send_error_email(request.recipient_email, "Analysis timed out", request.event_name)
        
    except Exception as e:
        logging.error(f"Task {task_id} failed: {str(e)}")
        await send_error_email(request.recipient_email, str(e), request.event_name)

async def analyze_columns_parallel(analyzer: Analyzer, df: pd.DataFrame, column_types: Dict[str, str]) -> Dict[str, Any]:
    """Analyze columns in parallel using ThreadPoolExecutor"""
    
    def analyze_single_column(column_data):
        column, col_type = column_data
        print(f"📊 Analyzing column: {column} (type: {col_type})")
        
        try:
            if col_type in ['numerical', 'rating']:
                analysis = analyzer.analyze_numerical_column(df, column)
                insights = analyzer.generate_column_insights(column, analysis)
                return column, {
                    'analysis': analysis,
                    'insights': insights.feedback,
                    'type': col_type
                }
                
            elif col_type == 'categorical':
                analysis = analyzer.analyze_categorical_column(df, column)
                insights = analyzer.generate_column_insights(column, analysis)
                return column, {
                    'analysis': analysis,
                    'insights': insights.feedback,
                    'type': col_type
                }
                
            elif col_type == 'text':
                analysis, all_text = analyzer.analyze_text_column(df, column)
                insights = analyzer.generate_column_insights(column, analysis, all_text)
                return column, {
                    'analysis': analysis,
                    'insights': insights,
                    'type': col_type
                }
                
        except Exception as e:
            logging.error(f"Error analyzing column {column}: {str(e)}")
            return column, {
                'error': str(e),
                'type': col_type
            }
    
    # Process columns in parallel
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=analyzer.config.MAX_WORKERS) as executor:
        column_items = list(column_types.items())
        
        # Submit all tasks
        tasks = [
            loop.run_in_executor(executor, analyze_single_column, item)
            for item in column_items
        ]
        
        # Wait for all results
        results_list = await asyncio.gather(*tasks)
        
        # Convert to dictionary
        results = dict(results_list)
        
    return results



async def validation_exception_handler(request: Request, exc: RequestValidationError):
    print(f"❌ Validation Error: {exc}")
    print(f"📋 Request body: {await request.body()}")
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation failed",
            "details": exc.errors(),
            "body": str(await request.body())
        }
    )


def print_terminal_separator(title):
    print("=" * 80)
    print(f"    {title}")
    print("=" * 80)

