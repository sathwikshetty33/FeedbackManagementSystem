from django.urls import path
from . import views
from .views import *
from authorization.views import *
urlpatterns = [
    path('student-login/',StudentLogin.as_view(),name='student-login'),
    path('teacher-login/',TeacherLogin.as_view(),name='teacher-login'),
    path('student-check/',CheckStudent.as_view(),name='student-check'),
    path('teacher-check/',CheckTeacher.as_view(),name='teacher-check'),
    path('admin-login/',SuperuserLogin.as_view(),name='admin-login'),
    path('student-check/',CheckSuperuser.as_view(),name='admin-check'),
    path('login/',Login.as_view(),name='login'),
    path('verify/',VerifyToken.as_view(),name='verify-token'),
    ]