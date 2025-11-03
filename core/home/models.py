from django.db import models
from django.contrib.auth.models import User

SEMESTER_CHOICES = [(str(i), f"Semester {i}") for i in range(1, 8)]
VISIBILITY_CHOICES = [
    ('1', 'Semester 1'),
    ('2', 'Semester 2'),
    ('3', 'Semester 3'),
    ('4', 'Semester 4'),
    ('5', 'Semester 5'),
    ('6', 'Semester 6'),
    ('7', 'Semester 7'),
    ('8', 'Semester 8'),
    ('anyone', 'Anyone'),
    ('teachers', 'Teachers'),
    ('students', 'Students')
]

class Student(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    semester = models.CharField(max_length=1, choices=SEMESTER_CHOICES)

    def __str__(self):
        return self.user.username

class Teacher(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)

    def __str__(self):
        return self.user.username

class Event(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField()
    start_time = models.DateTimeField()
    end_time = models.DateTimeField()
    visibility = models.CharField(max_length=10, choices=VISIBILITY_CHOICES)
    form_url = models.URLField()
    worksheet_url = models.URLField()
    created_by =models.ForeignKey(User, on_delete=models.CASCADE, related_name='created_events',null=True, blank=True)

    def __str__(self):
        return self.name

class EventGiven(models.Model):
    event = models.ForeignKey(Event, on_delete=models.CASCADE)
    user = models.ForeignKey(User, on_delete=models.CASCADE)

    def __str__(self):
        return f"{self.user.username} - {self.event.name}"
