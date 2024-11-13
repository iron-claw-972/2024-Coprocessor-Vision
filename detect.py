from roboflow import Roboflow
import os
import json

rf = Roboflow(api_key=os.environ["cRwQPP43BYHF5A6md5BI"])

#workspace = rf.workspace("cones-and-cubes")
workspace = rf.workspace("cones-and-cubes-o3dfe")

#projects = ["cones-and-cubes-o3dfe"]
projects = ["cones-and-cubes"]

def generate_and_train(project):
    
    rf_project = workspace.project(project)
    
    model = project.version(1).model
    
    # version_number = rf_project.generate_version()
    # project_item = workspace.project(project).version(version_number)

    project_item = workspace.project(project).version(model)
    
    project_item.train()
        

for project in projects:
    generate_and_train(projects[1])