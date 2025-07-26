from django.shortcuts import get_object_or_404, render
import uuid
from home.models import Event
import requests
# Create your views here.
def index(request):
    return render(request,'router/index.html')

def dashboard(request):
    return render(request,'router/dashboard.html')

def adminDashboard(request):
    return render(request,'router/adminDashboard.html')

def createevent(request,id=None):
    return render(request,'router/createEvent.html',{'id':id})
def about(request):
    return render(request,'router/about.html')
def chat_view(request):
    event_id = request.GET.get("event_id")
    if not event_id:
        print("No event ID provided in the request.")
        return render(request, "error.html", {"message": "No event ID provided."})

    event = get_object_or_404(Event, id=event_id)
    response_sheet_url = event.worksheet_url  # assuming this is a direct CSV URL
    desc = event.description if event.description else "No description provided."
    session_id = str(uuid.uuid4())
    print(f"Initializing chat session with ID: {session_id} for event: {event.name}")
    # Send to FastAPI
    fastapi_url = "http://localhost:8001/start_session"
    res = requests.post(fastapi_url, json={
        "session_id": session_id,
        "sheet_url": response_sheet_url,
        "description": desc,
    })

    if res.status_code != 200:
        print(f"Failed to initialize chat session: {res.status_code} - {res.text}")
        return render(request, "error.html", {"message": "Failed to initialize chat session."})

    return render(request, "router/chat.html", {
        "session_id": session_id,
        "event_id": event_id,
    })