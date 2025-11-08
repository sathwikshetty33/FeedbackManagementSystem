from django.urls import path
from . import views
from .views import *
urlpatterns = [
    path('',ListAvailableEvents.as_view(),name='all-events'),
    path('all-events/',views.all_events.as_view(),name='all-events'),
    path('events/',views.ListUserEvents.as_view(),name='user-events'),
    path('create-events/', EventListCreateAPIView.as_view(), name='event-list-create'),
    path('events/<int:pk>/', EventDetailAPIView.as_view(), name='event-detail'),
    path('claim-event/<int:event_id>/', ClaimEventAttendance.as_view(), name='claim-event'),
    path('attended-events/', ListAttendedEvents.as_view(), name='attended-events'),
    path('student-events/', GetAllStudentsEvents.as_view(), name='student-events'),
    path('teacher-events/', GetAllTeachersEvents.as_view(), name='teacher-events'),
    path('chat/<int:event_id>/', views.Chat.as_view(), name='chat'),
    ]