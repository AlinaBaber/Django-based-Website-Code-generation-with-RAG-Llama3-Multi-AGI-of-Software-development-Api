from django import forms
from .models import Inputfile

class InputfileForm(forms.ModelForm):
    class Meta:
        model = Inputfile
        fields = ['file']
        widgets = {
            'file': forms.ClearableFileInput(attrs={'class': 'form-control'}),
        }
