from rest_framework import views, status
from rest_framework.response import Response
from django.http import FileResponse, Http404
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from softwaredevelopmentapi.InputFile.serializers import InputfileSerializer
from softwaredevelopmentapi.InputFile.models import Inputfile
import os
from django.http import FileResponse
from rest_framework.parsers import MultiPartParser, FormParser
from softwaredevelopmentapi.SDAAgents.SDC import userinteraction_chatbot_query,business_analysis_agent,research_agent,software_test_agent,software_qa_agent,read_document_for_chatbot,software_developer_agent,software_architect_agent,software_developer_agent_openai
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import os
import shutil
from django.conf import settings
from django.http import FileResponse, Http404
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import json

class ResearcherView(APIView):
    parser_classes = (MultiPartParser, FormParser)  # Enables handling of multipart form data

    def post(self, request, *args, **kwargs):
        doc_serializer = InputfileSerializer(data=request.data)
        if doc_serializer.is_valid():
            doc_instance = doc_serializer.save()
            
            # Extracting details from dataset_instance
            folder_path = doc_instance.file.path.rsplit('/', 1)[0]
            file_name = doc_instance.file.name
            file_path = doc_instance.file.path
            response =research_agent(file_path)
            return Response({'Progress log': response}, status=status.HTTP_202_ACCEPTED)
        else:
            return Response(doc_serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class SoftwareProjectManagerView(APIView):

    def post(self, request):
        #task = request.data.get("task")
        projectid = request.data.get("projectid")
        if not projectid:
            return Response({"error": "No Project Information was provided"}, status=status.HTTP_400_BAD_REQUEST)
        else:
            response = business_analysis_agent(projectid)  # Assume this is your function to process queries
            return Response({'Progress log': response}, status=status.HTTP_200_OK)



class UserRequirementGatheringView(APIView):

    def post(self, request):
        query = request.data.get("query")
        print(query)
        key = request.data.get("key")
        projectid = request.data.get("projectid")
        if not query:
            return Response({"error": "No query provided"}, status=status.HTTP_400_BAD_REQUEST)
        else:
            response, newkey = userinteraction_chatbot_query(query, key,projectid) # Assume this is your function to process queries
            return Response({'Progress log': response, "key": newkey}, status=status.HTTP_200_OK)

class BusinessAnalysisView(APIView):
    parser_classes = (MultiPartParser, FormParser)  # Enables handling of multipart form data

    def post(self, request, *args, **kwargs):
        doc_serializer = InputfileSerializer(data=request.data)
        if doc_serializer.is_valid():
            doc_instance = doc_serializer.save()
            
            # Extracting details from dataset_instance
            folder_path = doc_instance.file.path.rsplit('/', 1)[0]
            file_name = doc_instance.file.name
            file_path = doc_instance.file.path
            project_id = doc_instance.projectid
            response = business_analysis_agent(file_path,project_id)
            return Response({'Progress log': response}, status=status.HTTP_202_ACCEPTED)
        else:
            return Response(doc_serializer.errors, status=status.HTTP_400_BAD_REQUEST)
class BusinessAnalysisDocumentView(APIView):

    def post(self, request):
        projectid = request.data.get("projectid")
        if not projectid:
            return Response({"error": "No Project Information was provided"}, status=status.HTTP_400_BAD_REQUEST)
        else:
            # Usage example
            file_path = f'Knowldgebase/BA/BADocumement{projectid}.docx'
            response = read_document_for_chatbot(file_path)
            #response = business_analysis_agent(projectid) # Assume this is your function to process queries
            return Response({'Text': response}, status=status.HTTP_200_OK)

class SoftwareArchitectView(APIView):

    def post(self, request):
        projectid = request.data.get("projectid")
        if not projectid:
            return Response({"error": "No Project Information was provided"}, status=status.HTTP_400_BAD_REQUEST)
        else:
            response = software_architect_agent(projectid) # Assume this is your function to process queries
            return Response({'Progress log': response}, status=status.HTTP_200_OK)

class SoftwareArchitectSRSDocumentView(APIView):

    def post(self, request):
        projectid = request.data.get("projectid")
        if not projectid:
            return Response({"error": "No Project Information was provided"}, status=status.HTTP_400_BAD_REQUEST)
        else:
            # Usage example
            file_path = f'Knowldgebase/SA/SRS{projectid}.docx'
            response = read_document_for_chatbot(file_path)
            #response = business_analysis_agent(projectid) # Assume this is your function to process queries
            return Response({'Text': response}, status=status.HTTP_200_OK)

class SoftwareArchitectDesignDocumentView(APIView):

    def post(self, request):
        projectid = request.data.get("projectid")
        if not projectid:
            return Response({"error": "No Project Information was provided"}, status=status.HTTP_400_BAD_REQUEST)
        else:
            # Usage example
            file_path = f'Knowldgebase/SA/Design{projectid}.docx'
            response = read_document_for_chatbot(file_path)
            #response = business_analysis_agent(projectid) # Assume this is your function to process queries
            return Response({'Text': response}, status=status.HTTP_200_OK)
            
class DevOpsView(APIView):

    def post(self, request):
        projectid = request.data.get("projectid")
        if not projectid:
            return Response({"error": "No Project Information was provided"}, status=status.HTTP_400_BAD_REQUEST)
        else:
            response = business_analysis_agent(projectid)  # Assume this is your function to process queries
            return Response({'Progress log': response}, status=status.HTTP_200_OK)

class SoftwareDeveloperView(APIView):

    def post(self, request):
        projectid = request.data.get("projectid")
        ui = request.data.get("ui")
        print("ui",ui)
        if not projectid:
            return Response({"error": "No Project Information was provided"}, status=status.HTTP_400_BAD_REQUEST)
        else:
                        # Store the ui value into a JSON file
            ui_data = {"ui": ui}
            try:
                with open('ui_data.json', 'w') as json_file:
                    json.dump(ui_data, json_file)
            except Exception as e:
                return Response({"error": f"Failed to save UI data: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
            response = software_developer_agent(projectid,ui_data)  # Assume this is your function to process queries
            return Response({'Progress log': response}, status=status.HTTP_200_OK)

class SoftwareDeveloperViewOpenAI(APIView):

    def post(self, request):
        projectid = request.data.get("projectid")
        model_name = request.data.get("model_name")
        if not projectid:
            return Response({"error": "No Project Information was provided"}, status=status.HTTP_400_BAD_REQUEST)
        else:
            response = software_developer_agent_openai(projectid,model_name)  # Assume this is your function to process queries
            return Response({'Progress log': response}, status=status.HTTP_200_OK)


class staticZipProjectDirectoryView(APIView):

    def post(self, request):
        projectid = request.data.get("projectid")
        if not projectid:
            return Response({"error": "No Project Information was provided"}, status=status.HTTP_400_BAD_REQUEST)
        
        base_dir = os.path.join(settings.BASE_DIR, f'Knowldgebase/SD/project/{projectid}')
        if not os.path.exists(base_dir):
            return Response({"error": "Project directory does not exist"}, status=status.HTTP_404_NOT_FOUND)
        
        zip_file_path = os.path.join(settings.BASE_DIR, f'Knowldgebase/SD/project/{projectid}.zip')
        
        # Create a zip file
        shutil.make_archive(zip_file_path.replace('.zip', ''), 'zip', base_dir)
        
        # Check if the zip file was created successfully
        if not os.path.exists(zip_file_path):
            return Response({"error": "Failed to create zip file"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # Send the zip file as a response
        try:
            zip_file = open(zip_file_path, 'rb')
            response = FileResponse(zip_file, content_type='application/zip')
            response['code_file'] = f'attachment; filename="{projectid}.zip"'
            return response
        except Exception as e:
            raise Http404(f"Error in sending file: {str(e)}")
            
class SoftwareDeveloperCodeDocumentView(APIView):

    def post(self, request):
        projectid = request.data.get("projectid")
        if not projectid:
            return Response({'Progress log': "No Project Information was provided"}, status=status.HTTP_400_BAD_REQUEST)
        else:
            # Usage example
            file_path =  f'Knowldgebase/SD/code_{projectid}.docx'
            response = read_document_for_chatbot(file_path)
            #response = business_analysis_agent(projectid) # Assume this is your function to process queries
            return Response({'Text': response}, status=status.HTTP_200_OK)
            
class SoftwareTesterView(APIView):

    def post(self, request):
        projectid = request.data.get("projectid")
        if not projectid:
            return Response({"error": "No Project Information was provided"}, status=status.HTTP_400_BAD_REQUEST)
        else:
            response = business_analysis_agent(projectid) # Assume this is your function to process queries
            return Response({'Progress log': response}, status=status.HTTP_200_OK)
            
class SoftwareTesterCodeDocumentView(APIView):

    def post(self, request):
        projectid = request.data.get("projectid")
        if not projectid:
            return Response({"error": "No Project Information was provided"}, status=status.HTTP_400_BAD_REQUEST)
        else:
            # Usage example
            file_path =  f'Knowldgebase/ST/UnitTest{projectid}.docx'
            response = read_document_for_chatbot(file_path)
            response = business_analysis_agent(projectid) # Assume this is your function to process queries
            return Response({'Progress log': response}, status=status.HTTP_200_OK)
            
class SoftwareQualityAssuranceView(APIView):

    def post(self, request):
        projectid = request.data.get("projectid")
        if not projectid:
            return Response({"error": "No Project Information was provided"}, status=status.HTTP_400_BAD_REQUEST)
        else:
            response = software_qa_agent(projectid)  # Assume this is your function to process queries
            return Response({'Progress log': response}, status=status.HTTP_200_OK)
class SoftwareQualityAssuranceDocumentView(APIView):

    def post(self, request):
        projectid = request.data.get("projectid")
        if not projectid:
            return Response({"error": "No Project Information was provided"}, status=status.HTTP_400_BAD_REQUEST)
        else:
            # Usage example
            file_path =  f'Knowldgebase/SQA/QA{projectid}.docx'
            response = read_document_for_chatbot(file_path)
            response = business_analysis_agent(projectid) # Assume this is your function to process queries
            return Response({'Progress log': response}, status=status.HTTP_200_OK)



class JSONFileUploadView(APIView):
    def post(self, request, *args, **kwargs):
        if 'file' not in request.FILES:
            return Response({"error": "No file uploaded"}, status=status.HTTP_400_BAD_REQUEST)
        
        file = request.FILES['file']
        try:
            json_data = json.load(file)
            # Process the JSON data here
            return Response({"message": "File processed successfully", "data": json_data}, status=status.HTTP_200_OK)
        except json.JSONDecodeError:
            return Response({"error": "Invalid JSON file"}, status=status.HTTP_400_BAD_REQUEST)
