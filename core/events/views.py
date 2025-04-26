from django.conf import settings
from rest_framework.response import Response
from rest_framework.views import APIView
import json
import os
from datetime import datetime, timedelta, timezone

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
        now = timezone.now()

        try:
            student = Student.objects.get(user=user)
            semester = student.semester
            events = Event.objects.filter(
                visibility__in=[semester, 'anyone'],
            )
        except Student.DoesNotExist:
            try:
                teacher = Teacher.objects.get(user=user)
                events = Event.objects.filter(
                    visibility__in=['teachers', 'anyone'],
                )
            except Teacher.DoesNotExist:
                return Response({"error": "User is neither a student nor a teacher"}, status=status.HTTP_400_BAD_REQUEST)
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
    permission_classes = [IsAuthenticated,IsAdminUser]
    def post(self, request):
        """
        Create a new event.
        Only admin users can create events.
        """
        if not request.user.is_superuser:
            return Response(
                {"detail": "You do not have permission to create events."},
                status=status.HTTP_403_FORBIDDEN
            )
        
        serializer = EventSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class EventDetailAPIView(APIView):
    """
    API view to retrieve, update or delete an event instance.
    """
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated,IsAdminUser]
    
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
        
        # Check if user has permission to view this event
        if not user.is_staff:  # Admin can view all events
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
        Only admin users can update events.
        """
        if not request.user.is_staff:
            return Response(
                {"detail": "You do not have permission to update events."},
                status=status.HTTP_403_FORBIDDEN
            )
        
        event = self.get_event(pk)
        serializer = EventSerializer(event, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    def delete(self, request, pk):
        """
        Delete an event.
        Only admin users can delete events.
        """
        if not request.user.is_staff:
            return Response(
                {"detail": "You do not have permission to delete events."},
                status=status.HTTP_403_FORBIDDEN
            )
        
        event = self.get_event(pk)
        event.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)
