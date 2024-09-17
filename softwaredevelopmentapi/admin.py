# admin.py


# @admin.register(DataSciencePipelineProcess)
# class DataSciencePipelineProcessAdmin(admin.ModelAdmin):
#     list_display = ['id', 'process', 'status', 'datetime']
#     list_filter = ['status', 'datetime']
#     search_fields = ['process', 'status']
#     readonly_fields = ['id', 'datetime']  # Assuming you want to keep track of when entries are created

#     fieldsets = (
#         (None, {
#             'fields': ('process', 'status')
#         }),
#         ('Date Information', {
#             'fields': ('datetime',),
#             'classes': ('collapse',)  # This section can be collapsed
#         }),
#     )

#     def view_related_results(self, obj):
#         """Custom function to link to related results, if needed."""
#         url = f"/admin/app_name/datasciencepipelineresult/?process__id__exact={obj.id}"
#         return format_html('<a href="{}">View Results</a>', url)

#     list_display += ('view_related_results',)  # Adding a link to view related results


# @admin.register(ProblemType)
# class ProblemTypeAdmin(admin.ModelAdmin):
#     list_display = ['id', 'type', 'description']
#     list_filter = ['type']  # Allows filtering by type in the admin interface
#     search_fields = ['type', 'description']  # Enables searching by type and description
#     fields = ['type', 'description']  # Fields to be edited in the admin form

#     def has_add_permission(self, request):
#         """Custom permission logic to decide if adding a problem type is permitted."""
#         # Here you might check if the user is a superuser or belongs to a specific group
#         return request.user.is_superuser

#     def has_change_permission(self, request, obj=None):
#         """Custom permission logic to decide if changing a problem type is permitted."""
#         return request.user.is_superuser

#     def has_delete_permission(self, request, obj=None):
#         """Custom permission logic to decide if deleting a problem type is permitted."""
#         return request.user.is_superuser


# @admin.register(DatasetandInput)
# class DatasetandInputAdmin(admin.ModelAdmin):
#     list_display = ['id', 'file_link', 'target_variable', 'problem_type_display', 'process_display', 'datetime_column']
#     list_filter = ['problem_type__type', 'process__status']
#     search_fields = ['target_variable', 'process__process', 'problem_type__type']
#     readonly_fields = ['file', 'target_variable', 'datetime_column']

#     def file_link(self, obj):
#         """Creates a clickable link to the dataset file."""
#         if obj.file:
#             return format_html('<a href="{0}" target="_blank">{1}</a>', obj.file.url, obj.file.name)
#         return "No file"
#     file_link.short_description = "Dataset File"

#     def process_display(self, obj):
#         """Display the related process more clearly."""
#         if obj.process:
#             return format_html('{} - {}', obj.process.id, obj.process.process)
#         return "No Process"
#     process_display.short_description = "Related Process"

#     def problem_type_display(self, obj):
#         """Display the related problem type more clearly."""
#         if obj.problem_type:
#             return format_html('{}', obj.problem_type.type)
#         return "No Problem Type"
#     problem_type_display.short_description = "Problem Type"

#     fieldsets = (
#         (None, {
#             'fields': ('file', 'target_variable', 'datetime_column')
#         }),
#         ('Related Information', {
#             'fields': ('process_display', 'problem_type_display'),
#             'description': 'Information regarding the related process and problem type.'
#         }),
#     )


# @admin.register(ProcessLog)
# class ProcessLogAdmin(admin.ModelAdmin):
#     list_display = ['id', 'process', 'timestamp', 'formatted_message']
#     list_filter = ['timestamp', 'process']
#     search_fields = ['message', 'process__id']
#     readonly_fields = ['id', 'process', 'timestamp', 'message']

#     def formatted_message(self, obj):
#         """Format the log message to display only the first 50 characters in the admin list view."""
#         return format_html('<div style="max-width: 800px; overflow: hidden; text-overflow: ellipsis;">{}</div>', obj.message[:50])
#     formatted_message.short_description = "Message"

#     fieldsets = (
#         (None, {
#             'fields': ('id', 'process', 'timestamp')
#         }),
#         ('Log Details', {
#             'fields': ('message',),
#             'description': "Full log message details."
#         }),
#     )


# @admin.register(DataSciencePipelineResult)
# class DataSciencePipelineResultAdmin(admin.ModelAdmin):
#     list_display = ['id', 'process', 'created_at', 'preview_result_data']
#     list_filter = ['created_at', 'process']
#     search_fields = ['process__id', 'result_data']
#     readonly_fields = ['id', 'process', 'created_at', 'result_data']
#     exclude = ['result_data']

#     def preview_result_data(self, obj):
#         """Creates a short preview of the JSON result data."""
#         result_data = obj.result_data
#         if result_data:
#             # Convert the JSON data to a string and truncate it for display purposes
#             result_data_str = str(result_data)
#             return format_html('<div style="max-width: 400px; overflow: hidden; text-overflow: ellipsis;">{}</div>', result_data_str[:150])
#         return None
#     preview_result_data.short_description = "Preview of Result Data"

#     fieldsets = (
#         (None, {
#             'fields': ('id', 'process', 'created_at')
#         }),
#         ('Results', {
#             'fields': ('preview_result_data',),
#             'description': "Preview of the JSON results stored for this process."
#         }),
#     )

# from django.contrib import admin
# from .models import *
