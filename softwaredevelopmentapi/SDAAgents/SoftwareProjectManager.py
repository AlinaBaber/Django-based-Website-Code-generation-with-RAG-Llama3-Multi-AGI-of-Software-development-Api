from datetime import date, timedelta
from django.shortcuts import get_object_or_404
from softwaredevelopmentapi.Project.models import Project
from softwaredevelopmentapi.Task.models import Task
from softwaredevelopmentapi.AIAgent.models import AIAgent
from softwaredevelopmentapi.ProgressLog.models import ProgressLog
from softwaredevelopmentapi.QualityAssuranceResults.models import QualityAssuranceResults
from softwaredevelopmentapi.SoftwareLifeCycle.models import SoftwareLifeCycle
from softwaredevelopmentapi.UseCases.models import UseCase
from softwaredevelopmentapi.UserStories.models import UserStory
from softwaredevelopmentapi.Client.models import Client
from softwaredevelopmentapi.RiskManagementPlan.models import RiskManagementPlan
from softwaredevelopmentapi.Requirement.models import Requirement
from .utils import send_email_to_stakeholders  # Assuming you have a utility function for sending emails
from softwaredevelopmentapi.model_loader import llm_model
class SoftwareProjectManager:
    def __init__(self, project_id=None):
        if project_id:
            self.project = get_object_or_404(Project, id=project_id)
        else:
            self.project = None


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

    # Planning Phase
    def define_project_scope(self, objectives, goals, deliverables, boundaries):
        self.project.objectives = objectives
        self.project.goals = goals
        self.project.deliverables = deliverables
        self.project.boundaries = boundaries
        self.project.save()

    def gather_requirements(self, stakeholder_interviews, requirement_documentation):
        requirements = Requirement.objects.create(
            project=self.project,
            stakeholder_interviews=stakeholder_interviews,
            requirement_documentation=requirement_documentation
        )
        return requirements

    def create_project_plan(self, schedule, tasks_responsibilities, milestones_deadlines):
        self.project.schedule = schedule
        self.project.tasks_responsibilities = tasks_responsibilities
        self.project.milestones_deadlines = milestones_deadlines
        self.project.save()

    def manage_risks(self, risks_identified, risk_impact_probability, risk_mitigation_strategies):
        risk_plan = RiskManagementPlan.objects.create(
            project=self.project,
            risks_identified=risks_identified,
            risk_impact_probability=risk_impact_probability,
            risk_mitigation_strategies=risk_mitigation_strategies
        )
        return risk_plan

    def plan_resources(self, resource_requirements, resource_allocation, resource_acquisition_plan):
        self.project.resource_requirements = resource_requirements
        self.project.resource_allocation = resource_allocation
        self.project.resource_acquisition_plan = resource_acquisition_plan
        self.project.save()

    def plan_budget(self, cost_estimates, budget_plan):
        self.project.cost_estimates = cost_estimates
        self.project.budget_plan = budget_plan
        self.project.save()

    def approve_budget(self, stakeholders):
        # Send the budget plan to stakeholders for approval
        send_email_to_stakeholders(stakeholders, "Budget Approval Request", self.project.budget_plan)

    # Execution Phase
    def kick_off_project(self):
        # Hold a kick-off meeting
        send_email_to_stakeholders(self.project.team.all(), "Project Kick-Off Meeting", "The project kick-off meeting will be held on...")

    def assign_tasks(self, ai_agent_id, task_id):
        ai_agent = get_object_or_404(AIAgent, id=ai_agent_id)
        task = get_object_or_404(Task, id=task_id)
        task.ai_agent = ai_agent
        task.save()

    def monitor_tasks(self):
        return Task.objects.filter(project=self.project)

    def adjust_tasks(self, task_id, new_details):
        task = get_object_or_404(Task, id=task_id)
        task.details = new_details
        task.save()

    def establish_communication(self, channels, status_meetings_schedule):
        self.project.communication_channels = channels
        self.project.status_meetings_schedule = status_meetings_schedule
        self.project.save()

    def provide_updates(self):
        send_email_to_stakeholders(self.project.stakeholders.all(), "Project Updates", "Here are the latest updates on the project...")

    def manage_quality(self, quality_standards):
        self.project.quality_standards = quality_standards
        self.project.save()

    def conduct_quality_checks(self):
        # Implement quality checks
        pass

    def ensure_quality(self):
        # Ensure deliverables meet quality standards
        pass

    def monitor_risks(self):
        return RiskManagementPlan.objects.filter(project=self.project)

    def implement_risk_mitigation(self):
        # Implement risk mitigation plans
        pass

    def adjust_risk_plans(self):
        # Adjust plans based on risk assessment
        pass

    # Monitoring and Controlling Phase
    def track_performance(self, kpis):
        self.project.kpis = kpis
        self.project.save()

    def manage_changes(self, change_requests):
        # Handle change requests
        pass

    def assess_changes(self, change_impact):
        # Assess the impact of changes
        pass

    def update_project_plan(self, changes):
        self.project.plan_changes = changes
        self.project.save()

    def control_budget(self, actual_spending):
        self.project.actual_spending = actual_spending
        self.project.save()

    def adjust_budget(self, new_budget):
        self.project.budget_plan = new_budget
        self.project.save()

    def report_budget_status(self):
        send_email_to_stakeholders(self.project.stakeholders.all(), "Budget Status", "Current budget status is...")

    def manage_resources(self, resource_utilization, resource_conflicts, resource_optimization):
        self.project.resource_utilization = resource_utilization
        self.project.resource_conflicts = resource_conflicts
        self.project.resource_optimization = resource_optimization
        self.project.save()

    # Closing Phase
    def close_project(self):
        # Confirm completion of all deliverables
        self.project.status = 'Completed'
        self.project.save()

    def confirm_stakeholder_approval(self):
        # Obtain stakeholder approval
        send_email_to_stakeholders(self.project.stakeholders.all(), "Project Completion Approval", "Please confirm the completion of the project...")

    def document_closure(self):
        # Document project closure
        pass

    def prepare_final_report(self):
        # Prepare final project report
        pass

    def debrief_team(self):
        # Conduct a debriefing session with the team
        send_email_to_stakeholders(self.project.team.all(), "Team Debriefing", "Let's discuss what went well and what didn't...")

    def celebrate_success(self):
        # Recognize team achievements and organize a project closure event
        pass

    def evaluate_post_project(self):
        # Conduct a post-project evaluation
        pass

    def review_outcomes(self):
        # Review project outcomes
        pass

    def identify_improvements(self):
        # Identify areas for improvement
        pass
