from rest_framework import serializers
from home.models import *

# class EventSerializer(serializers.ModelSerializer):
#     class Meta:
#         model = Event
#         fields = '__all__'  # or you can specify only required fields

from rest_framework import serializers
from home.models import Event  # Make sure Event is correctly imported

class EventSerializer(serializers.ModelSerializer):
    """
    Serializer for Event model.
    Automatically sets 'created_by' to the requesting user.
    """
    
    class Meta:
        model = Event
        # Remove 'created_by' from input fields
        fields = ['id', 'name', 'description', 'start_time', 'end_time', 
                  'visibility', 'form_url', 'worksheet_url']
    
    def validate(self, data):
        """
        Custom validation for event dates.
        Ensure that end_time is later than start_time.
        """
        if data['start_time'] >= data['end_time']:
            raise serializers.ValidationError(
                {"end_time": "End time must be later than start time."}
            )
        return data

    def create(self, validated_data):
        """
        Override create to set created_by from the request.
        """
        user = self.context['request'].user
        return Event.objects.create(created_by=user, **validated_data)
