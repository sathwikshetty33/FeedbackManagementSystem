# serializers.py
from rest_framework import serializers
from home.models import *

class EventSerializer(serializers.ModelSerializer):
    class Meta:
        model = Event
        fields = ['id', 'name', 'description', 'start_time', 'end_time', 
                  'visibility', 'form_url', 'worksheet_url']

class EventInsightRequestSerializer(serializers.Serializer):
    event_id = serializers.IntegerField()