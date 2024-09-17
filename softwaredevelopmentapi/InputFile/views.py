from django.urls import reverse_lazy
from django.views.generic import ListView, CreateView,DetailView, UpdateView, DeleteView
from .models import Inputfile
from .forms import InputfileForm
from rest_framework import generics
from .serializers import InputfileSerializer
from rest_framework import viewsets, filters
from .serializers import InputfileSerializer
from rest_framework.parsers import MultiPartParser, FormParser
class InputfileListView(ListView):
    model = Inputfile
    template_name = 'inputfile_list.html'
    context_object_name = 'inputfiles'

class InputfileCreateView(CreateView):
    model = Inputfile
    form_class = InputfileForm
    template_name = 'inputfile_form.html'
    success_url = reverse_lazy('inputfile-list')

class InputfileListCreateAPIView(generics.ListCreateAPIView):
    queryset = Inputfile.objects.all()
    serializer_class = InputfileSerializer


class InputfileDetailView(DetailView):
    model = Inputfile
    template_name = 'inputfile_detail.html'
    context_object_name = 'inputfile'

class InputfileUpdateView(UpdateView):
    model = Inputfile
    form_class = InputfileForm
    template_name = 'inputfile_form.html'
    success_url = reverse_lazy('inputfile-list')

class InputfileDeleteView(DeleteView):
    model = Inputfile
    template_name = 'inputfile_confirm_delete.html'
    success_url = reverse_lazy('inputfile-list')

class InputfileRetrieveUpdateDestroyAPIView(generics.RetrieveUpdateDestroyAPIView):
    queryset = Inputfile.objects.all()
    serializer_class = InputfileSerializer

class InputfileViewSet(viewsets.ModelViewSet):
    queryset = Inputfile.objects.all()
    serializer_class = InputfileSerializer
    filter_backends = (filters.SearchFilter, filters.OrderingFilter)
    search_fields = ['id', 'file']
    ordering_fields = ['id', 'file']
    parser_classes = (MultiPartParser, FormParser)  # Add this to handle file uploads

    def perform_create(self, serializer):
        serializer.save(file=self.request.data.get('file'))
