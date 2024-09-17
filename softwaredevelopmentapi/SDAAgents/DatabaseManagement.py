from datetime import date
from django.shortcuts import get_object_or_404
from softwaredevelopmentapi.Project.models import Project
from softwaredevelopmentapi.Task.models import Task
from softwaredevelopmentapi.ProgressLog.models import ProgressLog
from softwaredevelopmentapi.QualityAssuranceResults.models import QualityAssuranceResults
from softwaredevelopmentapi.SoftwareLifeCycle.models import SoftwareLifeCycle
from softwaredevelopmentapi.Client.models import Client
from softwaredevelopmentapi.AIAgent.models import AIAgent
from softwaredevelopmentapi.UserStories.models import UserStory
from softwaredevelopmentapi.UseCases.models import UseCase
from softwaredevelopmentapi.BusinessAnalysis.models import BusinessAnalysis
from softwaredevelopmentapi.SoftwarRequirementSpecifications.models import SoftwareRequirementSpecifications

class DatabaseManagement:
    def __init__(self, project_id=None):
        if project_id:
            self.project = get_object_or_404(Project, id=project_id)
        else:
            self.project = None

    @staticmethod
    def create_project(self,name, description, start_date, client_id, end_date=None, status='Pending'):
        client = get_object_or_404(Client, id=client_id)
        project = Project.objects.create(
            name=name,
            description=description,
            start_date=start_date,
            end_date=end_date,
            status=status,
            client=client
        )
        return project

    # Project Management
    def get_project_details(self):
        if not self.project:
            return None
        return {
            "name": self.project.name,
            "description": self.project.description,
            "start_date": self.project.start_date,
            "end_date": self.project.end_date,
            "status": self.project.status,
            "client": self.project.client.name,
            "ai_agents": [agent.name for agent in self.project.ai_agents.all()],
        }

    def update_project(self, name=None, description=None, start_date=None, end_date=None, status=None):
        if not self.project:
            return None
        if name:
            self.project.name = name
        if description:
            self.project.description = description
        if start_date:
            self.project.start_date = start_date
        if end_date:
            self.project.end_date = end_date
        if status:
            self.project.status = status
        self.project.save()
        return self.get_project_details()

    def add_ai_agent(self, agent_id):
        if not self.project:
            return None
        agent = get_object_or_404(AIAgent, id=agent_id)
        self.project.ai_agents.add(agent)
        return agent.name

    def remove_ai_agent(self, agent_id):
        if not self.project:
            return None
        agent = get_object_or_404(AIAgent, id=agent_id)
        self.project.ai_agents.remove(agent)
        return agent.name

    # Task Management
    def create_task(self, software_lifecycle_id, ai_agent_id, title, description, due_date):
        if not self.project:
            return None
        software_lifecycle = get_object_or_404(SoftwareLifeCycle, id=software_lifecycle_id)
        ai_agent = get_object_or_404(AIAgent, id=ai_agent_id)
        task = Task.objects.create(
            software_lifecycle=software_lifecycle,
            ai_agent=ai_agent,
            title=title,
            description=description,
            due_date=due_date
        )
        return task

    def update_task(self, task_id, title=None, description=None, ai_agent_id=None, due_date=None, status=None):
        if not self.project:
            return None
        task = get_object_or_404(Task, id=task_id)
        if title:
            task.title = title
        if description:
            task.description = description
        if ai_agent_id:
            ai_agent = get_object_or_404(AIAgent, id=ai_agent_id)
            task.ai_agent = ai_agent
        if due_date:
            task.due_date = due_date
        if status:
            task.status = status
        task.save()
        return task

    def delete_task(self, task_id):
        if not self.project:
            return None
        task = get_object_or_404(Task, id=task_id)
        task.delete()
        return task_id

    def get_tasks(self):
        if not self.project:
            return None
        return Task.objects.filter(software_lifecycle__project=self.project)

    # Progress Log Management
    def create_progress_log(self, progress, issues=None, next_steps=None):
        if not self.project:
            return None
        progress_log = ProgressLog.objects.create(
            project=self.project,
            date=date.today(),
            progress=progress,
            issues=issues,
            next_steps=next_steps
        )
        return progress_log

    def update_progress_log(self, log_id, progress=None, issues=None, next_steps=None):
        if not self.project:
            return None
        progress_log = get_object_or_404(ProgressLog, id=log_id)
        if progress:
            progress_log.progress = progress
        if issues:
            progress_log.issues = issues
        if next_steps:
            progress_log.next_steps = next_steps
        progress_log.save()
        return progress_log

    def delete_progress_log(self, log_id):
        if not self.project:
            return None
        progress_log = get_object_or_404(ProgressLog, id=log_id)
        progress_log.delete()
        return log_id

    def get_progress_logs(self):
        if not self.project:
            return None
        return ProgressLog.objects.filter(project=self.project)

    # Quality Assurance Management
    def create_quality_assurance_result(self, test_type, title, document, description):
        if not self.project:
            return None
        qa_result = QualityAssuranceResults.objects.create(
            project=self.project,
            test_type=test_type,
            title=title,
            document=document,
            description=description
        )
        return qa_result

    def update_quality_assurance_result(self, result_id, test_type=None, title=None, document=None, description=None):
        if not self.project:
            return None
        qa_result = get_object_or_404(QualityAssuranceResults, id=result_id)
        if test_type:
            qa_result.test_type = test_type
        if title:
            qa_result.title = title
        if document:
            qa_result.document = document
        if description:
            qa_result.description = description
        qa_result.save()
        return qa_result

    def delete_quality_assurance_result(self, result_id):
        if not self.project:
            return None
        qa_result = get_object_or_404(QualityAssuranceResults, id=result_id)
        qa_result.delete()
        return result_id

    def get_quality_assurance_results(self):
        if not self.project:
            return None
        return QualityAssuranceResults.objects.filter(project=self.project)

    # AI Agent Management
    def get_ai_agents(self):
        if not self.project:
            return None
        return AIAgent.objects.filter(tasks__software_lifecycle__project=self.project).distinct()

    # Software Lifecycle Management
    def get_software_lifecycles(self):
        if not self.project:
            return None
        return SoftwareLifeCycle.objects.filter(project=self.project)

    def create_software_lifecycle(self, lifecycle_type, start_date, end_date=None, description=None):
        if not self.project:
            return None
        software_lifecycle = SoftwareLifeCycle.objects.create(
            project=self.project,
            lifecycle_type=lifecycle_type,
            start_date=start_date,
            end_date=end_date,
            description=description
        )
        return software_lifecycle

    def update_software_lifecycle(self, lifecycle_id, lifecycle_type=None, start_date=None, end_date=None, description=None):
        if not self.project:
            return None
        software_lifecycle = get_object_or_404(SoftwareLifeCycle, id=lifecycle_id)
        if lifecycle_type:
            software_lifecycle.lifecycle_type = lifecycle_type
        if start_date:
            software_lifecycle.start_date = start_date
        if end_date:
            software_lifecycle.end_date = end_date
        if description:
            software_lifecycle.description = description
        software_lifecycle.save()
        return software_lifecycle

    def delete_software_lifecycle(self, lifecycle_id):
        if not self.project:
            return None
        software_lifecycle = get_object_or_404(SoftwareLifeCycle, id=lifecycle_id)
        software_lifecycle.delete()
        return lifecycle_id

    # Use Case Management
    def create_use_case(self, title, description):
        if not self.project:
            return None
        use_case = UseCase.objects.create(
            project=self.project,
            title=title,
            description=description
        )
        return use_case

    def update_use_case(self, use_case_id, title=None, description=None):
        if not self.project:
            return None
        use_case = get_object_or_404(UseCase, id=use_case_id)
        if title:
            use_case.title = title
        if description:
            use_case.description = description
        use_case.save()
        return use_case

    def delete_use_case(self, use_case_id):
        if not self.project:
            return None
        use_case = get_object_or_404(UseCase, id=use_case_id)
        use_case.delete()
        return use_case_id

    def get_use_cases(self):
        if not self.project:
            return None
        return UseCase.objects.filter(project=self.project)

    # User Story Management
    def create_user_story(self, use_case_id, title, description):
        if not self.project:
            return None
        use_case = get_object_or_404(UseCase, id=use_case_id)
        user_story = UserStory.objects.create(
            use_case=use_case,
            title=title,
            description=description
        )
        return user_story

    def update_user_story(self, user_story_id, title=None, description=None):
        if not self.project:
            return None
        user_story = get_object_or_404(UserStory, id=user_story_id)
        if title:
            user_story.title = title
        if description:
            user_story.description = description
        user_story.save()
        return user_story

    def delete_user_story(self, user_story_id):
        if not self.project:
            return None
        user_story = get_object_or_404(UserStory, id=user_story_id)
        user_story.delete()
        return user_story_id

    def get_user_stories(self):
        if not self.project:
            return None
        return UserStory.objects.filter(use_case__project=self.project)

    # Document Management
    def create_document(self, title, file, description=None):
        if not self.project:
            return None
        document = Document.objects.create(
            project=self.project,
            title=title,
            file=file,
            description=description
        )
        return document

    def update_document(self, document_id, title=None, file=None, description=None):
        if not self.project:
            return None
        document = get_object_or_404(Document, id=document_id)
        if title:
            document.title = title
        if file:
            document.file = file
        if description:
            document.description = description
        document.save()
        return document

    def delete_document(self, document_id):
        if not self.project:
            return None
        document = get_object_or_404(Document, id=document_id)
        document.delete()
        return document_id

    def get_documents(self):
        if not self.project:
            return None
        return Document.objects.filter(project=self.project)

    # Additional Modules Management
    def create_business_analysis(self, title, description):
        if not self.project:
            return None
        analysis = BusinessAnalysis.objects.create(
            project=self.project,
            title=title,
            description=description
        )
        return analysis

    def update_business_analysis(self, analysis_id, title=None, description=None):
        if not self.project:
            return None
        analysis = get_object_or_404(BusinessAnalysis, id=analysis_id)
        if title:
            analysis.title = title
        if description:
            analysis.description = description
        analysis.save()
        return analysis

    def delete_business_analysis(self, analysis_id):
        if not self.project:
            return None
        analysis = get_object_or_404(BusinessAnalysis, id=analysis_id)
        analysis.delete()
        return analysis_id

    def get_business_analyses(self):
        if not self.project:
            return None
        return BusinessAnalysis.objects.filter(project=self.project)

    def create_design(self, title, file, description=None):
        if not self.project:
            return None
        design = Design.objects.create(
            project=self.project,
            title=title,
            file=file,
            description=description
        )
        return design

    def update_design(self, design_id, title=None, file=None, description=None):
        if not self.project:
            return None
        design = get_object_or_404(Design, id=design_id)
        if title:
            design.title = title
        if file:
            design.file = file
        if description:
            design.description = description
        design.save()
        return design

    def delete_design(self, design_id):
        if not self.project:
            return None
        design = get_object_or_404(Design, id=design_id)
        design.delete()
        return design_id

    def get_designs(self):
        if not self.project:
            return None
        return Design.objects.filter(project=self.project)

    def create_risk_management_plan(self, title, description):
        if not self.project:
            return None
        risk_management_plan = RiskManagementPlan.objects.create(
            project=self.project,
            title=title,
            description=description
        )
        return risk_management_plan

    def update_risk_management_plan(self, plan_id, title=None, description=None):
        if not self.project:
            return None
        plan = get_object_or_404(RiskManagementPlan, id=plan_id)
        if title:
            plan.title = title
        if description:
            plan.description = description
        plan.save()
        return plan

    def delete_risk_management_plan(self, plan_id):
        if not self.project:
            return None
        plan = get_object_or_404(RiskManagementPlan, id=plan_id)
        plan.delete()
        return plan_id

    def get_risk_management_plans(self):
        if not self.project:
            return None
        return RiskManagementPlan.objects.filter(project=self.project)

    def create_system_architecture(self, title, file, description=None):
        if not self.project:
            return None
        architecture = SystemArchitecture.objects.create(
            project=self.project,
            title=title,
            file=file,
            description=description
        )
        return architecture

    def update_system_architecture(self, architecture_id, title=None, file=None, description=None):
        if not self.project:
            return None
        architecture = get_object_or_404(SystemArchitecture, id=architecture_id)
        if title:
            architecture.title = title
        if file:
            architecture.file = file
        if description:
            architecture.description = description
        architecture.save()
        return architecture

    def delete_system_architecture(self, architecture_id):
        if not self.project:
            return None
        architecture = get_object_or_404(SystemArchitecture, id=architecture_id)
        architecture.delete()
        return architecture_id

    def get_system_architectures(self):
        if not self.project:
            return None
        return SystemArchitecture.objects.filter(project=self.project)