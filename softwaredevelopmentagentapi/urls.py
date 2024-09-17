from django.urls import path, include
from rest_framework.routers import DefaultRouter

from softwaredevelopmentapi.views import UserRequirementGatheringView,BusinessAnalysisView,SoftwareArchitectView,SoftwareDeveloperView,BusinessAnalysisDocumentView,SoftwareArchitectSRSDocumentView,SoftwareArchitectDesignDocumentView,SoftwareDeveloperCodeDocumentView,staticZipProjectDirectoryView,SoftwareDeveloperViewOpenAI,JSONFileUploadView
from softwaredevelopmentapi.InputFile.views import InputfileViewSet
router = DefaultRouter()
router.register(r'InputFile', InputfileViewSet)
#router.register(r'DatasetandInput', DatasetandInputViewSet)
#router.register(r'ProcessLog', ProcessLogViewSet)
#router.register(r'DataSciencePipelineResult', DataSciencePipelineResultViewSet)
#router.register(r'ProblemType', ProblemTypeViewSet)
#router.register(r'api',RunPipelineView)

urlpatterns = [
    path('', include(router.urls)),
    path('user-requirement-gathering/', UserRequirementGatheringView.as_view(), name='user-requirement-gathering'),
    path('business-analysis/', BusinessAnalysisView.as_view(), name='business-analysis'),
    path('business-analysis-document/', BusinessAnalysisDocumentView.as_view(), name='business-analysis-document'),
    path('software-architect/', SoftwareArchitectView.as_view(), name='software-architect'),
    path('software-architect-srs/', SoftwareArchitectSRSDocumentView.as_view(), name='software-architect-srs'),
    path('software-architect-design/', SoftwareArchitectDesignDocumentView.as_view(), name='software-architect-design'),
    path('software-developer/', SoftwareDeveloperView.as_view(), name='software-developer'),
    path('software-developer-openai/', SoftwareDeveloperViewOpenAI.as_view(), name='software-developer-openai'),
    path('software-developer-code/', staticZipProjectDirectoryView.as_view(), name='software-developer-code-static'),
    path('software-developer-codedoc/', SoftwareDeveloperCodeDocumentView.as_view(), name='software-developer-code-doc'),
    path('upload-json/', JSONFileUploadView.as_view(), name='upload-json')

]