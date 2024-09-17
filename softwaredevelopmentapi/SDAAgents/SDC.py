from softwaredevelopmentapi.SDAAgents.UserInteractionBot import UserInteractionBot
from softwaredevelopmentapi.SDAAgents.BusinessAnalyst import BusinessAnalystAgent
from softwaredevelopmentapi.SDAAgents.Researcher import ResearcherAgent
from softwaredevelopmentapi.SDAAgents.SoftwareArchitect import SoftwareArchitectAgent
from softwaredevelopmentapi.SDAAgents.SoftwareDeveloper import SoftwareDeveloperAgent
from softwaredevelopmentapi.SDAAgents.SoftwareTester import SoftwareTesterAgent
from softwaredevelopmentapi.SDAAgents.SoftwareQualityAssurance import  SoftwareQualityAssuranceAgent
#from softwaredevelopmentapi.SDAAgents.SoftwareDeveloper import DjangoModelCodeGenerator
from softwaredevelopmentapi.SDAAgents.CodeSeparator import extract_and_create_html_files ,extract_and_create_css_file
from softwaredevelopmentapi.SDAAgents.OpenAIChat import generatecode_openai
from softwaredevelopmentapi.SDAAgents.OpenAIChat import generatecode_openai
import os

def is_folder_empty(folder_path):
    # Check if the given path is a directory
    if not os.path.isdir(folder_path):
        raise ValueError(f"The path {folder_path} is not a directory.")
    
    # Check if the directory is empty
    if not os.listdir(folder_path):
        return True
    else:
        return False
def userinteraction_chatbot_query(query,key,projectid):
    bot =UserInteractionBot()
    output_path ='Knowldgebase/client/conversation.docx'
    f'Knowldgebase/Research/ResearchDocument{projectid}.docx'
    response,key =bot.run_bot(query,key,output_path,projectid)
    return response,key


def business_analysis_agent_(projectid):
    # Example usage:
    agent = BusinessAnalystAgent()
    input_path=f'Knowldgebase/client/conversation{projectid}.docx'
    output_path=f'Knowldgebase/BA/BADocumement{projectid}.docx'
    #projectname="Online Ecommerce Website on Django Platform"
    response = agent.business_analyst_agent(input_path, output_path)
    print(response)
    return response
    
def business_analysis_agent(file_path,projectid):
    # Example usage:
    agent = BusinessAnalystAgent()
    #input_path=f'Knowldgebase/client/conversation{projectid}.docx'
    output_path=f'Knowldgebase/BA/BADocumement{projectid}.docx'
    #projectname="Online Ecommerce Website on Django Platform"
    response = agent.business_analyst_agent(file_path, output_path,projectid)
    print(response)
    return response
    
def research_agent(input_path):
    agent = ResearcherAgent()
    #input_path = 'Knowldgebase/SA/SRS.docx'
    output_path = f'Knowldgebase/Research/ResearchDocument{projectid}.docx'
    response = agent.researcher_agent(input_path, output_path)
    print(response)
    return response


def software_architect_agent(projectid):
    agent = SoftwareArchitectAgent()
    output_path = f'Knowldgebase/SA/SRS{projectid}.docx'
    projectname="Online Ecommerce Website on Django Platform"
    result = agent.software_architect_agent(f'Knowldgebase/BA/BADocumement{projectid}.docx', output_path,projectid)
    print(result)

    #projectname="Online Ecommerce Website on Django Platform"
    output_path = f'Knowldgebase/SA/Design{projectid}.docx'
    result = agent.software_architect_agent_design(f'Knowldgebase/SA/SRS{projectid}.docx', output_path,projectid)
    print(result)
    response = "SRS and Design has been completed."
    return response

def is_folder_empty(folder_path):
    # Check if the given path is a directory
    if not os.path.isdir(folder_path):
        raise ValueError(f"The path {folder_path} is not a directory.")
    
    # Check if the directory is empty
    if not os.listdir(folder_path):
        return True
    else:
        return False
def software_developer_agent(projectid,ui_data):
    agent = SoftwareDeveloperAgent()
    #projectname="Online Ecommerce Website on Django Platform"
    output_path = f'Knowldgebase/SD/code_{projectid}.docx'
    base_dir= f'Knowldgebase/SD/project/{projectid}'
    os.makedirs(base_dir, exist_ok=True)
    input_path=f'Knowldgebase/SA/SQL_Database{projectid}.docx'
    #response = agent.software_developer_agent_full(input_path, output_path,base_dir,projectid)
    response = agent.software_developer_agent_full(input_path, output_path, base_dir,projectid)
    if is_folder_empty(base_dir):
        input_path=f'Knowldgebase/SA/SRS{projectid}.docx'
        response = agent.software_developer_agent_full(input_path, output_path, base_dir,projectid)
        
    print("Files have been saved successfully.")
    print(response)
    return response
def software_developer_agentf(projectid,ui_data):
    agent = SoftwareDeveloperAgent()
    #projectname="Online Ecommerce Website on Django Platform"
    output_path = f'Knowldgebase/SD/code_{projectid}.docx'
    base_dir= f'Knowldgebase/SD/project/{projectid}'
    os.makedirs(base_dir, exist_ok=True)
    input_path=f'Knowldgebase/SA/SQL_Database{projectid}.docx'
    #response = agent.software_developer_agent_full(input_path, output_path,base_dir,projectid)
    response = agent.software_developer_agent_template(input_path, output_path, base_dir,projectid,ui_data)
    # Example usage
    docx_file_path = output_path
    output_directory = base_dir

    extract_and_create_html_files(docx_file_path, output_directory)
    docx_path=f'Knowldgebase/SD/styles_{projectid}.docx'
    extract_and_create_css_file(docx_path,base_dir)
    #code_extractor.separate_code()
    #code_extractor.save_to_files()
    
    # Instantiate the class and save the code to files
    # code_extractor = CodeExtractor(docx_file_path)
    # code_extractor.save_code_to_files(output_directory)
    print("Files have been saved successfully.")
    print(response)
    return response

def software_developer_agent_openai(projectid,model_name):
    #agent = SoftwareDeveloperAgent()
    #projectname="Online Ecommerce Website on Django Platform"
    output_path = f'Knowldgebase/SD/code_{projectid}.docx'
    base_dir= f'Knowldgebase/SD/project/{projectid}'
    os.makedirs(base_dir, exist_ok=True)
    input_path=f'Knowldgebase/SA/HTMLPages{projectid}.docx'
    #response = agent.software_developer_agent_full(input_path, output_path,base_dir,projectid)
    response = generatecode_openai(input_path, output_path, projectid,model_name)
    # Example usage
    docx_file_path = output_path
    output_directory = base_dir
    
    # Instantiate the class and save the code to files
    code_extractor = CodeExtractor(docx_file_path)
    code_extractor.save_code_to_files(output_directory)
    print("Files have been saved successfully.")
    print(response)
    return response

##########
# def software_developer_agent(projectid):
#     input_path = f'Knowldgebase/SA/SQL_Database{projectid}.docx'
#     output_path = f'Knowldgebase/SD/Code/project_{projectid}/'
#     generator = DjangoModelCodeGenerator(
#     doc_path=input_path,
#     model_id="meta-llama/Meta-Llama-3-8B-Instruct",
#     embedding_model="sentence-transformers/all-MiniLM-L6-v2"
#     )
#     generator.load_document()
#     generator.extract_model_names()
#     generator.set_up_embeddings()
#     generator.set_up_language_model()
#     generator.create_qa_chain()
#     prompts, answers = generator.generate_code()
#     response = generator.display_and_save_results(prompts, answers)
#     return response
######

def software_test_agent():
    agent = SoftwareTesterAgent()
    projectname="Online Ecommerce Website on Django Platform"
    output_path = f'Knowldgebase/ST/UnitTest{projectid}.docx'
    base_dir='Project'
    response = agent.software_test_agent_unittests(f'Knowldgebase/SD/Code{projectid}.docx', output_path,base_dir)
    print(response)
    return response


def software_qa_agent():
    agent = SoftwareQualityAssuranceAgent()
    projectname="Online Ecommerce Website on Django Platform"
    output_path = f'Knowldgebase/SQA/QA{projectid}.docx'
    base_dir='Project'
    response = agent.software_test_agent_unittests(f'Knowldgebase/SA/SRS{projectid}.docx', output_path,base_dir)
    print(response)
    return response

from docx import Document
import os

def read_document_for_chatbot(file_path):
    """
    Reads a .docx document from the specified file path and returns its content formatted for display in a chatbot.

    Args:
    - file_path (str): The path to the .docx document.

    Returns:
    - str: The formatted content of the document.
    """
    if not os.path.exists(file_path):
        return "The specified document does not exist."

    try:
        doc = Document(file_path)
        content = []

        for para in doc.paragraphs:
            content.append(para.text)

        formatted_content = "\n".join(content)
        return formatted_content

    except Exception as e:
        return f"An error occurred while reading the document: {e}"
