from django.shortcuts import render

# Create your views here.
def index(request):
    return render(request,'router/index.html')

def dashboard(request):
    return render(request,'router/dashboard.html')

def adminDashboard(request):
    return render(request,'router/adminDashboard.html')

def createevent(request,id=None):
    return render(request,'router/createEvent.html',{'id':id})
