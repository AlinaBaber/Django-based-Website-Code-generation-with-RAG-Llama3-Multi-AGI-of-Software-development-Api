from rest_framework import serializers
from .models import Inputfile

class InputfileSerializer(serializers.ModelSerializer):
    class Meta:
        model = Inputfile
        fields = ['file', 'projectid']

