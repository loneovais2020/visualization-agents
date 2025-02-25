import os
import PyPDF2
import docx
import pandas as pd
from prompts import *
from langchain_experimental.agents import create_pandas_dataframe_agent
import pandas as pd
from langchain_google_vertexai import ChatVertexAI
import io
from crewai import Agent, Task, Crew, LLM
from crewai_tools import CodeInterpreterTool, SerperDevTool
from dotenv import load_dotenv
load_dotenv()




# Initialize the tool for internet searching capabilities
internet_search_tool = SerperDevTool()
run_codes = CodeInterpreterTool(unsafe_mode=True)


llm = LLM(
# model="vertex_ai/gemini-1.5-flash-001", temperature=0.6
# model="gemini/gemini-1.5-pro-latest", temperature=0.6
model="vertex_ai/gemini-2.0-flash-001", temperature=0.9
)



llm_model = ChatVertexAI(model_name= "gemini-1.5-flash-001")



def read_file_content(filepath, purpose='summary'):
    # Check if the file exists
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"The file {filepath} does not exist.")
    
    # Get the file extension
    _, file_extension = os.path.splitext(filepath)
    file_extension = file_extension.lower()
    
    if file_extension == '.pdf':
        return read_pdf(filepath, purpose)
    elif file_extension == '.docx':
        return read_docx(filepath, purpose)
    elif file_extension in ['.xlsx', '.csv']:
        return read_spreadsheet(filepath, purpose)
    elif file_extension == '.txt':
        return read_txt(filepath, purpose)
    else:
        raise ValueError("Unsupported file type. Allowed types are: .pdf, .docx, .xlsx, .csv, .txt")

def read_pdf(filepath, purpose):
    with open(filepath, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        num_pages = len(reader.pages)
        if purpose == 'summary':
            pages_to_read = num_pages // 2  # Read first 50% of pages
        else:
            pages_to_read = num_pages  # Read all pages
        content = []
        for page_num in range(pages_to_read):
            page = reader.pages[page_num]
            content.append(page.extract_text())
    return '\n'.join(content)

def read_docx(filepath, purpose):
    doc = docx.Document(filepath)
    paragraphs = doc.paragraphs
    num_paragraphs = len(paragraphs)
    if purpose == 'summary':
        paragraphs_to_read = num_paragraphs // 2  # Read first 50% of paragraphs
    else:
        paragraphs_to_read = num_paragraphs  # Read all paragraphs
    content = [paragraphs[i].text for i in range(paragraphs_to_read)]
    return '\n'.join(content)

def read_spreadsheet(filepath, purpose):
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    else:
        df = pd.read_excel(filepath)
    
    num_rows = len(df)
    if purpose == 'summary':
        rows_to_read = num_rows // 5  # Calculate 20% of the total rows
        df_subset = df.head(rows_to_read)  # Select the first 20% of rows
    else:
        df_subset = df  # Select all rows
    
    # Convert the DataFrame subset to a string
    df_string = df_subset.to_string(index=False)
    
    return df_string

def read_txt(filepath, purpose):
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.readlines()
    
    num_lines = len(content)
    if purpose == 'summary':
        lines_to_read = num_lines // 2  # Read first 50% of lines
    else:
        lines_to_read = num_lines  # Read all lines
    
    return ''.join(content[:lines_to_read])



file_selection_agent = Agent(
    llm=llm,
    role="File Selection Expert",
    goal="Analyze user queries and return relevant file names based on file content summaries.",
    backstory="A highly skilled data indexing expert who can efficiently match user needs with available document summaries.",
    allow_code_execution=False  # No need for code execution, just text processing
)


file_matching_task = Task(
    description="""Compare the user query with file content summaries and return a relevant file name that can be used for the user query.
    If their is no suitable file available, then  output ''
    The file summaries are: {file_summaries}
    The user query is: {user_query}
    
    Be precise in your matching, considering both explicit and implicit data requirements in the query.""",
    agent=file_selection_agent,
    expected_output="Filename that matches the user query or ''" 
)


file_selection_crew = Crew(
    agents=[file_selection_agent],
    tasks=[file_matching_task],
    verbose=True
)









# New Data Processing Agent
# dataset_processor_agent = Agent(
#     llm=llm,
#     role="Data Processor",
#     goal="Transform and prepare data for analysis by generating processed datasets that can be directly used for analysis and visualization.",
#     backstory="""You are a specialized data processing expert who excels at transforming raw data into analysis-ready formats.
#     You understand data structures deeply and can efficiently clean, transform, and restructure data to make the analyst's job easier.
#     Your expertise is in creating intermediate data representations that are optimized for specific analysis tasks.""",
#     allow_code_execution=True,
#     tools=[CodeInterpreterTool(unsafe_mode=True)]
# )

# # New Data Processing Task
# dataset_processing_task = Task(
#     description="""Your task is to process and transform the extracted data into a format that's optimized for the specific analysis needed.
    
#     Process steps:
#     1. Load the raw data using proper error handling with try-except blocks
#     2. Understand what exact data transformations are needed based on the analysis requirements
#     3. Perform necessary data cleaning, including:
#        - Handling missing values
#        - Converting data types
#        - Normalizing or scaling values if needed
#        - Filtering out irrelevant records
#        - Creating derived features if beneficial
#     4. Reshape the data if needed (pivot, melt, etc.)
#     5. Document all transformations performed
#     6. Provide the processed data in a format that can be directly used by the analysis agent
    
#     Code quality requirements:
#     - Use proper error handling with try-except blocks
#     - Include clear comments explaining each transformation step
#     - Print summaries of the data before and after processing
#     - Use efficient methods for data transformation
#     - Document any assumptions made during processing
    
#     The user query is: {user_query}
#     The raw data is located at: {file_path}
#     """,
#     agent=dataset_processor_agent,
#     expected_output="""A processed dataset with the following information:
#     1. The processed data in a suitable format
#     2. Documentation of all transformations applied
#     3. Summary statistics of the processed data
#     4. Any relevant insights or issues discovered during processing"""
# )

# dataset_data_analysis_crew = Crew(
#     agents=[dataset_processor_agent, dataset_analyst_agent],
#     tasks=[dataset_processing_task, dataset_data_analysis_task],
#     verbose= True
# )


# data_preparation_agent = Agent(
#     llm=llm,
#     role="Data Preparation Specialist",
#     goal="Interpret user queries and extract relevant data from provided file content with careful consideration of requirements.",
#     backstory="""An expert in data extraction and preparation, skilled at understanding user requirements and processing raw data accordingly.
#     You excel at interpreting the nuances of data requests and preparing exactly what's needed for analysis.""",
#     allow_code_execution=False
# )

# data_preparation_task = Task(
#     description="""Interpret the user query and extract the necessary data from the provided file content.
    
#     Process:
#     1. Carefully analyze the user query to understand:
#        - What specific data elements are needed
#        - What transformations might be required
#        - Whether the query requires analysis, visualization, or both
    
#     2. Extract and structure the relevant data from the file content
    
#     The user query is: {user_query}
#     The file content is: {file_content}""",
#     agent=data_preparation_agent,
#     expected_output="Processed data ready for analysis with clear structure and documentation of any transformations performed."
# )




# data_analysis_crew = Crew(
#     agents=[data_preparation_agent,dataset_processor_agent],
#     tasks=[data_preparation_task, data_preparation_task],
#     verbose=True
# )





# File Data Extraction Agent - Extracts relevant data from the file content
file_data_extraction_agent = Agent(
    llm=llm,
    role="File Data Extraction Specialist",
    goal="Extract relevant data from provided file content based on the user query.",
    backstory="""You are an expert in processing file content and extracting meaningful data from it.
    You analyze the user query, identify relevant data, and structure it for further processing into a CSV file.""",
    allow_code_execution=False
)

# Task for File Data Extraction Agent
file_data_extraction_task = Task(
    description="""Your task is to extract relevant data from the provided file content based on the user's query.
    
    Process:
    1. Analyze the user query to determine the specific data required
    2. Read and parse the file content
    3. Extract and organize relevant data in a structured format
    4. Ensure the extracted data is:
       - Relevant to the user's query
       - Well-structured and formatted for CSV conversion
       - Ready for processing by the CSV generation agent
    5. Provide the structured data to the CSV data processor agent

    The user query is: {user_query}\n
    The file content is: {file_content}\n
    """,
    agent=file_data_extraction_agent,
    expected_output="""Extracted data related to the user's query, including:
    1. The structured or semi-structured data from the file
    2. Any necessary context for understanding the data"""
)

# CSV Data Processor Agent - Converts extracted data into a CSV file
csv_file_processor_agent = Agent(
    llm=llm,
    role="CSV Data Processor",
    goal="Transform extracted file data into a structured CSV file.",
    backstory="""You are an expert in data processing and transformation. Your main responsibility is to take the extracted data 
    and convert it into a clean, well-structured CSV file, ensuring proper formatting for analysis.""",
)

# Task for CSV Data Processor Agent
csv_file_processing_task = Task(
    description="""Your task is to process the extracted data from file content and output it as a CSV file.
    
    Process steps:
    1. Load the structured data extracted by the file data extraction agent
    2. Clean and structure the data, including:
       - Ensuring uniform column naming
       - Handling missing values appropriately
       - Converting unstructured data into structured format (if necessary)
    3. Generate a CSV file with the structured data
    4. Output only the CSV file, without additional summaries or explanations

    The user query is: {user_query}
    """,
    agent=csv_file_processor_agent,
    expected_output="""A CSV file containing:
    1. Structured data extracted from the file
    2. Properly formatted column names
    3. No additional text or explanation, just the CSV file"""
)

# Create the file processing crew
file_processing_crew = Crew(
    agents=[file_data_extraction_agent, csv_file_processor_agent],
    tasks=[file_data_extraction_task, csv_file_processing_task],
    verbose=True
)

# Internet Search Agent - Collects data from the web
internet_search_agent = Agent(
    llm=llm,
    role="Internet Research Specialist",
    goal="Gather relevant data and information from the internet based on user queries",
    backstory="""You are an expert internet researcher skilled at finding and extracting relevant information from online sources.
    You analyze user queries to determine what specific information is needed and retrieve structured and unstructured data from reliable sources.""",
    allow_code_execution=False,
    tools=[internet_search_tool]
)

# Task for Internet Search Agent
internet_search_task = Task(
    description="""Your task is to search the internet for data and information relevant to the user's query.
    
    Process:
    1. Analyze the user query to determine the specific data required
    2. Formulate effective search queries to retrieve the most relevant information
    3. Collect and organize the retrieved data in a structured format
    4. Ensure the collected data is:
       - Relevant to the user's query
       - Structured when possible
       - From reliable sources
       - Ready for CSV conversion
    5. Provide the raw data to the data processing agent

    The user query is: {user_query}
    """,
    agent=internet_search_agent,
    expected_output="""Collected data related to the user's query, including:
    1. The raw data in a structured or semi-structured format
    2. Sources of the information
    3. Any context necessary for understanding the data"""
)

# CSV Data Processor Agent - Converts collected data into a CSV file
csv_data_processor_agent = Agent(
    llm=llm,
    role="CSV Data Processor",
    goal="Transform internet search data into a structured CSV file.",
    backstory="""You are an expert in data processing and transformation. Your main responsibility is to take the collected raw data 
    and convert it into a well-structured CSV file. You ensure that the output is clean, organized, and properly formatted for analysis.""",
)

# Task for CSV Data Processor Agent
csv_data_processing_task = Task(
    description="""Your task is to process the data collected from internet searches and output it as a CSV file.
    
    Process steps:
    1. Load the raw data from the internet research agent
    2. Clean and structure the data, including:
       - Ensuring uniform column naming
       - Handling missing values appropriately
       - Converting unstructured data into structured format (if necessary)
    3. Generate a CSV file with the structured data
    4. Output only the CSV file, without additional summaries or explanations

    The user query is: {user_query}
    """,
    agent=csv_data_processor_agent,
    expected_output="""A CSV file containing:
    1. Structured data from the internet search
    2. Properly formatted column names
    3. No additional text or explanation, just the CSV file"""
)



# Create the internet search and analysis crew
internet_search_analysis_crew = Crew(
    agents=[internet_search_agent, csv_data_processor_agent],
    tasks=[internet_search_task, csv_data_processing_task],
    verbose=True
)



from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from pydantic import BaseModel
from uuid import uuid4
from datetime import datetime
from pymongo import MongoClient
import os
import shutil
from bson import ObjectId
from typing import List




# Allowed file extensions
ALLOWED_EXTENSIONS = {"xlsx", "csv", "txt", "pdf", "docx"}

def is_allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


app = FastAPI()

# MongoDB connection
client = MongoClient("mongodb+srv://loneovais2019:cmZIilbGCgyqoZPc@dev.gvpfh.mongodb.net/")
db = client["project_db"]
projects_collection = db["projects"]

# Directory to save uploaded files
BASE_DIR = os.getcwd()
FILES_DIR = os.path.join(BASE_DIR, "FILES")
os.makedirs(FILES_DIR, exist_ok=True)

class Project(BaseModel):
    project_name: str
    user_id: str
@app.post("/create-project/")
async def create_project(project: Project):
    project_id = str(uuid4())
    creation_time = datetime.utcnow()
    
    # Prepare project data with additional fields
    project_data = {
        "project_name": project.project_name,
        "project_id": project_id,
        "user_id": project.user_id,
        "creation_time": creation_time,
        "uploaded_files": [],
        "chat_history": []
    }
    
    # Save project details to MongoDB
    projects_collection.insert_one(project_data)
    
    # Create project directory
    project_dir = os.path.join(FILES_DIR, f"{project.project_name}_{project_id}", "uploaded_files")
    os.makedirs(project_dir, exist_ok=True)
    
    return {
        "project_id": project_id,
        "project_name": project.project_name,
        "user_id": project.user_id,
        "creation_time": creation_time
    }

@app.post("/upload-file/")
async def upload_file(project_id: str = Form(...), file: UploadFile = File(...)):
    # Retrieve project details from MongoDB
    project = projects_collection.find_one({"project_id": project_id})
    if not project:
        raise HTTPException(status_code=404, detail="Project not found.")
    
    if not is_allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="File type not allowed.")
    
    # Define the path to save the uploaded file
    project_dir = os.path.join(FILES_DIR, f"{project['project_name']}_{project_id}", "uploaded_files")
    os.makedirs(project_dir, exist_ok=True)
    file_path = os.path.join(project_dir, file.filename)
    
    # Save the uploaded file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Read the content of the file
    file_content = read_file_content(file_path)
    
    # Generate a prompt based on the file content
    file_summary_prompt = file_summary_generation_prompt(file_content)
    
    # Invoke the model to get the summary
    summary = llm_model.invoke(file_summary_prompt)
    
    # Create a dictionary with filename and summary
    file_info = {
        "filename": file.filename,
        "summary": summary.content
    }
    
    # Update the project's uploaded_files in MongoDB
    projects_collection.update_one(
        {"project_id": project_id},
        {"$push": {"uploaded_files": file_info}}
    )
    
    return file_info


def extract_csv_content(text):
    """
    Extracts CSV content surrounded by ```csv and ``` markers from a string.
    
    Args:
        text (str): The input string containing CSV content marked with ```csv and ``` tags
    
    Returns:
        str: The extracted CSV content or empty string if no match found
    """
    import re
    
    # Define pattern to match content between ```csv and ``` markers
    pattern = r"```csv\s*(.*?)\s*```"
    
    # Search for the pattern with re.DOTALL flag to match across multiple lines
    match = re.search(pattern, text, re.DOTALL)
    
    # Return the matched content or empty string if no match
    if match:
        return match.group(1)
    else:
        return ""



class ChatRequest(BaseModel):
    project_id: str
    user_query: str

@app.post("/chat/")
async def chat(request: ChatRequest):
    # Retrieve project details from MongoDB
    project = projects_collection.find_one({"project_id": request.project_id})
    if not project:
        raise HTTPException(status_code=404, detail="Project not found.")
    
    # Load the uploaded_files list of the project
    file_summary_dict_list = project.get("uploaded_files", [])
    
    inputs = {
        'user_query': request.user_query,
        'file_summaries': file_summary_dict_list
    }
    
    # Select relevant files based on the user query
    file_name = file_selection_crew.kickoff(inputs=inputs)
    file_name = str(file_name.raw)

    print("----------------------------------------")
    print(type(file_name))
    print(file_name)
    print("----------------------------------------")
    
    # if not file_name:

        # return {"message": "No file available for this query."}


    # if file_name == "":
    #     print("CALLING INTERNET CREW........")
    #     response = internet_search_analysis_crew.kickoff({"user_query": request.user_query})

    # else:

        
        
        
        # for filename in files_to_use:
        #     filename = filename[1]
        #     print("----------------------------------------")
        #     print(type(filename))
        #     print(filename)
        #     print("----------------------------------------")
    file_extension = file_name.rsplit('.', 1)[-1].lower()
    print("----------------------------------------")
    print(f"File extension is {file_extension}")
    print("----------------------------------------")

    project_folder = f"{project['project_name']}_{request.project_id}"
    file_path = os.path.join(FILES_DIR, project_folder, "uploaded_files", file_name)
    charts_folder = os.path.join(FILES_DIR, project_folder, "created_charts")
    
    if file_extension in ["csv", "xlsx"]:
        # Process CSV or XLSX files
        if file_extension == "csv":
            df = pd.read_csv(file_path)
        elif file_extension =="xlsx":
            df = pd.read_excel(file_path)
    
    elif file_extension in ["pdf", "txt", "docx"]:
            print("""Entering ["pdf", "txt", "docx"] Block""")
            file_content = file_processing_crew.kickoff({"user_query": request.user_query, "file_content": read_file_content(file_path)})
            formatted_csv_content = extract_csv_content(str(file_content.raw))
            df = pd.read_csv(io.StringIO(formatted_csv_content))


    else:
        print("Entering FILE EXTENSION '' BLOCK.")
        internet_content = internet_search_analysis_crew.kickoff({"user_query": request.user_query})
        formatted_csv_content = extract_csv_content(str(internet_content.raw))
        df = pd.read_csv(io.StringIO(formatted_csv_content))




    print(df.head())

    agent_executor = create_pandas_dataframe_agent(
    llm_model,
    df,
    # extra_tools=,
    # agent_type="tool-calling",
    allow_dangerous_code= True,
    verbose=True
)
        
    response = agent_executor.invoke(data_analysis_prompt(request.user_query, charts_folder))
    print(response)
        # response = dataset_data_analysis_crew.kickoff({
        #     'user_query': request.user_query,
        #     'file_path': file_path,
        #     "charts_folder":charts_folder
        # })
    # elif file_extension in ["pdf", "txt", "docx"]:
    #     # Process PDF, TXT, or DOCX files
    #     file_content = read_file_content(file_path)
    #     response = data_analysis_crew.kickoff({
    #         'user_query': request.user_query,
    #         'file_content': file_content,
    #         "charts_folder": charts_folder
    #     })

    # # elif file_extension =="":
        
    # else:
    #     print("CALLING INTERNET CREW........")
    #     response = internet_search_analysis_crew.kickoff({"user_query": request.user_query, "charts_folder": charts_folder})
    #     # response = {"message": f"Unsupported file type: {file_extension}"}
    
    print(response['output'])

    response = {
        "filename": file_name,
        "user_query": request.user_query,
        # "response": response.raw
        "response": response['output']
    }

    # Update the project's uploaded_files in MongoDB
    projects_collection.update_one(
        {"project_id": request.project_id},
        {"$push": {"chat_history": response}}
    )

    return response


class ProjectModel(BaseModel):
    project_id: str
    project_name: str
    creation_time: datetime
    uploaded_files: List[dict] = []
    chat_history: List[dict] = []
    user_id: str



@app.get("/project/{project_id}", response_model=ProjectModel)
async def get_project(project_id: str):
    # Retrieve the project from MongoDB
    project = projects_collection.find_one({"project_id": project_id})
    if not project:
        raise HTTPException(status_code=404, detail="Project not found.")
    
    # Convert ObjectId to string if present
    if "_id" in project:
        project["_id"] = str(project["_id"])
    
    return project