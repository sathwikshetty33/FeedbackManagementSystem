from django.urls import path
from . import views
from .views import *
urlpatterns = [
    path('',views.dashboard,name='user-dashboard'),
    path('index/',views.index,name="index"),
    path('admin-dashboard/',views.adminDashboard,name="admin-dashboard"),
    path('admin-create/',views.createevent,name='admin-create'),
    path('admin-create/<int:id>/',views.createevent,name='admin-create'),
    path('about/',views.about,name='about'),
    path('chat/', views.chat_view, name='chat_view'),
    ]