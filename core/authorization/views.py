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
from .serializers import *
from rest_framework.permissions import AllowAny
from rest_framework.authtoken.models import Token
from rest_framework.authentication import TokenAuthentication
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
import random
import string


class StudentLogin(APIView):
        authentication_classes = []
        permission_classes = [AllowAny]
        def post(self, request):
            data = request.data
            serializer=LoginSerializer(data=data)
            if not serializer.is_valid():
                return Response({"some error":serializer.errors},status=status.HTTP_400_BAD_REQUEST)
            username = serializer.data['username']
            password = serializer.data['password']
            us = authenticate(username=username,password=password)
            if us is None:
                return  Response({
                    "error" : "Invalid username and password"
                },status=status.HTTP_401_UNAUTHORIZED)
            try:
                d = Student.objects.get(user=us)
            except Student.DoesNotExist:
                return  Response({
                    "error" : "You are not a student"
                },status=status.HTTP_401_UNAUTHORIZED)
            token,_ = Token.objects.get_or_create(user=us)
            return Response({
                "token" : token.key,
                "hospId" : d.id,
            },status=status.HTTP_200_OK)

class TeacherLogin(APIView):
        authentication_classes = []
        permission_classes = [AllowAny]
        def post(self, request):
            data = request.data
            serializer=LoginSerializer(data=data)
            if not serializer.is_valid():
                return Response({"some error":serializer.errors},status=status.HTTP_400_BAD_REQUEST)
            username = serializer.data['username']
            password = serializer.data['password']
            us = authenticate(username=username,password=password)
            if us is None:
                return  Response({
                    "error" : "Invalid username and password"
                },status=status.HTTP_401_UNAUTHORIZED)
            try:
                d = Teacher.objects.get(user=us)
            except Teacher.DoesNotExist:
                return  Response({
                    "error" : "You are not a teacher"
                },status=status.HTTP_401_UNAUTHORIZED)
            token,_ = Token.objects.get_or_create(user=us)
            return Response({
                "token" : token.key,
                "hospId" : d.id,
            },status=status.HTTP_200_OK)
        
class CheckStudent(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user
        try:
            Student.objects.get(user=user)
            return Response({"user_type": "Student"}, status=status.HTTP_200_OK)
        except:
            return Response({"error": "User is not a student"}, status=status.HTTP_401_UNAUTHORIZED)

class CheckTeacher(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user
        try:
            Teacher.objects.get(user=user)
            return Response({"user_type": "Teacher"}, status=status.HTTP_200_OK)
        except:
            return Response({"error": "User is not a Teacher"}, status=status.HTTP_401_UNAUTHORIZED)
        

class SuperuserLogin(APIView):
    authentication_classes = []
    permission_classes = [AllowAny]
    
    def post(self, request):
        data = request.data
        serializer = LoginSerializer(data=data)
        if not serializer.is_valid():
            return Response({"error": serializer.errors}, status=status.HTTP_400_BAD_REQUEST)
        
        username = serializer.data['username']
        password = serializer.data['password']
        user = authenticate(username=username, password=password)
        
        if user is None:
            return Response({
                "error": "Invalid username and password"
            }, status=status.HTTP_401_UNAUTHORIZED)
            
        if not user.is_superuser:
            return Response({
                "error": "You are not a superuser"
            }, status=status.HTTP_401_UNAUTHORIZED)
            
        token, _ = Token.objects.get_or_create(user=user)
        return Response({
            "token": token.key,
            "userId": user.id,
        }, status=status.HTTP_200_OK)


class CheckSuperuser(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user
        if user.is_superuser:
            return Response({"user_type": "Superuser"}, status=status.HTTP_200_OK)
        else:
            return Response({"error": "User is not a superuser"}, status=status.HTTP_401_UNAUTHORIZED)

class Login(APIView):
    permission_classes = [AllowAny]
    def post(self, request):
        data = request.data
        serializer = LoginSerializer(data=data)
        if not serializer.is_valid():
            return Response({"error": serializer.errors}, status=status.HTTP_400_BAD_REQUEST)
        
        username = serializer.data['username']
        password = serializer.data['password']
        user = authenticate(username=username, password=password)
        
        if user is None:
            return Response({
                "error": "Invalid username and password"
            }, status=status.HTTP_401_UNAUTHORIZED)
        if Student.objects.filter(user=user).exists():
            user_type = "Student"
        elif Teacher.objects.filter(user=user).exists():
            user_type = "Teacher"
        elif user.is_superuser:
            user_type = "Superuser"
        token, _ = Token.objects.get_or_create(user=user)
        return Response({
            "token": token.key,
            "userId": user.id,
            "user_type": user_type
        }, status=status.HTTP_200_OK)
    
class VerifyToken(APIView):
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        user = request.user
        if Student.objects.filter(user=user).exists():
            user_type = "Student"
        elif Teacher.objects.filter(user=user).exists():
            user_type = "Teacher"
        elif user.is_superuser:
            user_type = "Superuser"
            
        return Response({
            "userId": user.id,
            "username": user.username,
            "user_type": user_type
        }, status=status.HTTP_200_OK)