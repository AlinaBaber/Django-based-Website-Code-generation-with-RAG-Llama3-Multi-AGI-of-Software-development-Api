from django.db import models

from django.db import models
import json

class Inputfile(models.Model):
    id = models.AutoField(primary_key=True)
    file = models.FileField(upload_to='inputdocs/')
    projectid = models.IntegerField(null=True, blank=True)
    
    def __str__(self):
        return f"Inputfile {self.id} for Project {self.project_id}"


    def save(self, *args, **kwargs):
        super(Inputfile, self).save(*args, **kwargs)



    def get_file_path(self):
        return self.file.path