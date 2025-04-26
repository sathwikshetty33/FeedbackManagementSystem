from django.contrib import admin
from .models import *
# Register your models here.
admin.site.register(Student)
admin.site.register(Teacher)
admin.site.register(Event)
admin.site.register(EventGiven)