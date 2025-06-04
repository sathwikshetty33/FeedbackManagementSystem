# myapp/permissions.py
from rest_framework.permissions import BasePermission
from home.models import *
class IsTeacher(BasePermission):
    """
    Allows access only to users who are Teacher instances.
    """
    def has_permission(self, request, view):
        return bool(
            request.user and
            request.user.is_authenticated and
            Teacher.objects.filter(user=request.user).exists()
        )
