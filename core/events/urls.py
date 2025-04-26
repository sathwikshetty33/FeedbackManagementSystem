from django.urls import path
from . import views
from .views import *
urlpatterns = [
    path('',ListAvailableEvents.as_view(),name='all-events'),
    path('events/',views.ListUserEvents.as_view(),name='user-events'),
     path('events/', EventListCreateAPIView.as_view(), name='event-list-create'),
    path('events/<int:pk>/', EventDetailAPIView.as_view(), name='event-detail'),
    ]