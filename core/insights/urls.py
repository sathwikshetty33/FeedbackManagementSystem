# urls.py
from django.urls import path
from . import views

urlpatterns = [
    # Your existing URL patterns
    path('generate-insights/', views.GenerateInsightsView.as_view(), name='generate-insights'),
]