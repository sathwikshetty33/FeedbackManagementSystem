from django.conf import settings
from rest_framework.response import Response
from rest_framework.views import APIView
import json
import os
from datetime import datetime, timedelta, timezone
from events.permission import *
from rest_framework.permissions import IsAuthenticated, IsAdminUser
from django.shortcuts import render,redirect
from home.models import *
from django.contrib.auth import authenticate
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from django.shortcuts import get_object_or_404
from django.conf import settings
from django.db.models import Q
from rest_framework.permissions import AllowAny
from rest_framework.authtoken.models import Token
from rest_framework.authentication import TokenAuthentication
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
import random
import string
from django.utils import timezone
from .serializers import *
import uuid
import requests
class all_events(APIView):
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated, IsAdminUser | IsTeacher]  # Combine permissions properly (DRF 3.9+)

    def get(self, request):
        user = request.user
        
        # Check if user is staff (admin)
        if user.is_staff:
            events = Event.objects.all()
        else:
            events = Event.objects.filter(created_by=user)

        serializer = EventSerializer(events, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)
class ListAvailableEvents(APIView):
    permission_classes = [AllowAny]

    def get(self, request):
        now = timezone.now()
        events = Event.objects.filter(start_time__lte=now, end_time__gte=now,visibility='anyone')
        serializer = EventSerializer(events, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)

class ListUserEvents(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user
        
        try:
            student = Student.objects.get(user=user)
            semester = student.semester
            # Get events visible to this student's semester or anyone
            events = Event.objects.filter(
                visibility__in=[semester, 'anyone'],
            )
        except Student.DoesNotExist:
            try:
                teacher = Teacher.objects.get(user=user)
                # Get events visible to teachers or anyone
                events = Event.objects.filter(
                    visibility__in=['teachers', 'anyone'],
                )
            except Teacher.DoesNotExist:
                return Response({"error": "User is neither a student nor a teacher"}, status=status.HTTP_400_BAD_REQUEST)
        
        # Filter out events that the user has already claimed
        claimed_events = EventGiven.objects.filter(user=user).values_list('event_id', flat=True)
        events = events.exclude(id__in=claimed_events)
        
        serializer = EventSerializer(events, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)
    
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated, IsAdminUser
from rest_framework.authentication import TokenAuthentication
from django.shortcuts import get_object_or_404
from .serializers import EventSerializer
class EventListCreateAPIView(APIView):
    """
    API view to list all events or create a new event.
    """
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated,IsAdminUser | IsTeacher]
    def post(self, request):
        """
        Create a new event.
        Only admin users can create events.
        """

        serializer = EventSerializer(data=request.data, context={'request': request})
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class EventDetailAPIView(APIView):
    """
    API view to retrieve, update or delete an event instance.
    """
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated, IsAdminUser | IsTeacher]

    def get_event(self, pk):
        """Helper method to get event or return 404"""
        return get_object_or_404(Event, pk=pk)

    def get(self, request, pk):
        """
        Retrieve an event by id.
        Access rules are the same as for list view.
        """
        event = self.get_event(pk)
        user = request.user

        # Admin can view all events
        if not user.is_staff:
            if hasattr(user, 'teacher'):
                if event.visibility not in ['teachers', 'anyone']:
                    return Response(
                        {"detail": "You do not have permission to view this event."},
                        status=status.HTTP_403_FORBIDDEN
                    )
            elif hasattr(user, 'student'):
                student_semester = str(user.student.semester)
                if event.visibility not in [student_semester, 'anyone']:
                    return Response(
                        {"detail": "You do not have permission to view this event."},
                        status=status.HTTP_403_FORBIDDEN
                    )
            else:
                if event.visibility != 'anyone':
                    return Response(
                        {"detail": "You do not have permission to view this event."},
                        status=status.HTTP_403_FORBIDDEN
                    )

        serializer = EventSerializer(event)
        return Response(serializer.data)

    def put(self, request, pk):
        """
        Update an event.
        Admins can update any event.
        Teachers can update only events they created.
        """
        event = self.get_event(pk)
        user = request.user

        if not user.is_staff:
            # Check teacher ownership
            if not hasattr(user, 'teacher') or event.created_by != user:
                return Response(
                    {"detail": "You do not have permission to update this event."},
                    status=status.HTTP_403_FORBIDDEN
                )

        serializer = EventSerializer(event, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, pk):
        """
        Delete an event.
        Admins can delete any event.
        Teachers can delete only events they created.
        """
        event = self.get_event(pk)
        user = request.user

        if not user.is_staff:
            # Check teacher ownership
            if not hasattr(user, 'teacher') or event.created_by != user:
                return Response(
                    {"detail": "You do not have permission to delete this event."},
                    status=status.HTTP_403_FORBIDDEN
                )

        event.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


class ClaimEventAttendance(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, event_id):
        user = request.user
        
        try:
            event = Event.objects.get(id=event_id)
            
            # Check if user has already claimed this event
            if EventGiven.objects.filter(user=user, event=event).exists():
                return Response({"error": "You have already attended this event"}, status=status.HTTP_400_BAD_REQUEST)
            
            # Check if user is eligible to claim this event
            try:
                student = Student.objects.get(user=user)
                if event.visibility not in [student.semester, 'anyone']:
                    return Response({"error": "You are not eligible for this event"}, status=status.HTTP_403_FORBIDDEN)
            except Student.DoesNotExist:
                try:
                    Teacher.objects.get(user=user)
                    if event.visibility not in ['teachers', 'anyone']:
                        return Response({"error": "You are not eligible for this event"}, status=status.HTTP_403_FORBIDDEN)
                except Teacher.DoesNotExist:
                    return Response({"error": "User is neither a student nor a teacher"}, status=status.HTTP_400_BAD_REQUEST)
            
            # Create the attendance record
            EventGiven.objects.create(user=user, event=event)
            
            return Response({"success": "Event attendance recorded successfully"}, status=status.HTTP_201_CREATED)
            
        except Event.DoesNotExist:
            return Response({"error": "Event not found"}, status=status.HTTP_404_NOT_FOUND)
        
class ListAttendedEvents(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user
        
        # Get all events that the user has claimed
        attended_events_ids = EventGiven.objects.filter(user=user).values_list('event_id', flat=True)
        attended_events = Event.objects.filter(id__in=attended_events_ids)
        
        serializer = EventSerializer(attended_events, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)
    

class GetAllStudentsEvents(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        try:
            # Ensure the user has a student profile
            student = request.user.student
        except Exception:
            return Response({"error": "Student profile not found."}, status=status.HTTP_404_NOT_FOUND)

        # Fetch events visible to this student
        events = Event.objects.filter(
            visibility__in=['anyone', str(student.semester),'students']
        ).order_by('-start_time')

        # Build the response list
        response_data = []
        for event in events:
            is_marked = EventGiven.objects.filter(event=event, user=request.user).exists()
            response_data.append({
                "id": event.id,
                "name": event.name,
                "description": event.description,
                "start_time": event.start_time,
                "end_time": event.end_time,
                "visibility": event.visibility,
                "form_url": event.form_url,
                "worksheet_url": event.worksheet_url,
                "created_by": event.created_by.username if event.created_by else None,
                "is_marked": is_marked
            })

        return Response(response_data, status=status.HTTP_200_OK)

class GetAllTeachersEvents(APIView):
    """
    API view to list all events created by the authenticated teacher.
    """
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated, IsTeacher]

    def get(self, request):
        try:
            # Ensure the user has a teacher profile
            teacher = request.user.teacher
        except Exception:
            return Response({"error": "Teacher profile not found."}, status=status.HTTP_404_NOT_FOUND)

        # Fetch events visible to this student
        events = Event.objects.filter(
            visibility__in=['anyone', 'teachers']
        ).order_by('-start_time')

        # Build the response list
        response_data = []
        for event in events:
            is_marked = EventGiven.objects.filter(event=event, user=request.user).exists()
            response_data.append({
                "id": event.id,
                "name": event.name,
                "description": event.description,
                "start_time": event.start_time,
                "end_time": event.end_time,
                "visibility": event.visibility,
                "form_url": event.form_url,
                "worksheet_url": event.worksheet_url,
                "created_by": event.created_by.username if event.created_by else None,
                "is_marked": is_marked
            })

        return Response(response_data, status=status.HTTP_200_OK)

class Chat(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request,event_id):
        
        if not event_id:
            print("No event ID provided in the request.")
            return Response({"message": "No event ID provided."}, status=status.HTTP_400_BAD_REQUEST)

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
            return Response({"error": "Failed to initialize chat session."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response({
            "session_id": session_id,
            "event_id": event_id,
        },status=status.HTTP_200_OK)    