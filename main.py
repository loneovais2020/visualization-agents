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
from fastapi.middleware.cors import CORSMiddleware
import json
from fastapi.staticfiles import StaticFiles
import shutil

load_dotenv()




# Initialize the tool for internet searching capabilities
internet_search_tool = SerperDevTool()
run_codes = CodeInterpreterTool(unsafe_mode=True)


llm = LLM(
# model="vertex_ai/gemini-1.5-flash-001", temperature=0.6
# model="gemini/gemini-1.5-pro-latest", temperature=0.6
model="vertex_ai/gemini-2.0-flash-001", temperature=0.9
)



llm_model = ChatVertexAI(model_name= "gemini-2.0-flash-001")



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

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],  # Explicitly list allowed methods
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,  # Cache preflight requests for 1 hour
)


# MongoDB connection
client = MongoClient("mongodb+srv://loneovais2019:cmZIilbGCgyqoZPc@dev.gvpfh.mongodb.net/")
db = client["project_db"]
projects_collection = db["projects"]
users_collection = db["users"]

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
    
    # Check if file already exists in project's uploaded_files
    if project.get("uploaded_files"):
        for existing_file in project["uploaded_files"]:
            if existing_file["filename"] == file.filename:
                raise HTTPException(
                    status_code=400, 
                    detail={
                        "message": "File already exists",
                        "filename": file.filename
                    }
                )
    
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

    # Add file upload entry to chat history
    chat_entry = {
        "user_query": f"[FILE]{file.filename}",
        "response": {
            "response": summary.content,
            "created_charts": []
        }
    }

    # Update the project's chat_history in MongoDB
    projects_collection.update_one(
        {"project_id": project_id},
        {"$push": {"chat_history": chat_entry}}
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
    
def extract_json_content(text):
    """
    Extracts JSON content surrounded by ```json and ``` markers from a string
    and converts it to a Python dictionary. If no markers are found, attempts
    to convert the entire string to a dictionary.
    
    Args:
        text (str): The input string containing JSON content (with or without markers)
    
    Returns:
        dict: The extracted JSON content converted to a dictionary, or empty dict if conversion fails
    """
    import re
    import json
    import ast
    
    print(f"Input text: {text}")
    
    # Check if the text is wrapped in ```json and ``` markers
    if text.strip().startswith("```json") and text.strip().endswith("```"):
        print("Detected JSON code block markers")
        
        # Define pattern to match content between ```json and ``` markers
        pattern = r"```json\s*(.*?)\s*```"
        
        # Search for the pattern with re.DOTALL flag to match across multiple lines
        match = re.search(pattern, text, re.DOTALL)
        
        # Process the matched content
        if match:
            json_str = match.group(1)
            print(f"Extracted content: {json_str}")
        else:
            print("Failed to extract content with regex")
            return {}
    else:
        print("No JSON code block markers found, treating entire text as JSON/dict")
        json_str = text
    
    # Try to convert to dict
    try:
        # First try regular JSON parsing
        result = json.loads(json_str)
        print(f"Successfully parsed with json.loads(): {result}")
        return result
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        try:
            # If JSON parsing fails, try evaluating as Python literal
            result = ast.literal_eval(json_str)
            print(f"Successfully parsed with ast.literal_eval(): {result}")
            return result
        except (SyntaxError, ValueError) as e:
            print(f"Python literal evaluation error: {e}")
            # Return empty dict if both methods fail
            return {}



class ChatRequest(BaseModel):
    project_id: str
    user_query: str


def _handle_error(error) -> str:
    return str(error)[:50]

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
    handle_parsing_errors=_handle_error,
    verbose=True
)
        
    response = agent_executor.invoke(data_analysis_prompt(request.user_query, charts_folder))
    
    # Extract just the text response from the output
    response_text = response['output']
    
    # Get charts from the created_charts folder
    created_charts = []
    if os.path.exists(charts_folder):
        created_charts = [os.path.join(charts_folder, f) for f in os.listdir(charts_folder) 
                         if os.path.isfile(os.path.join(charts_folder, f))]
        

    print(f"Raw response from model is: {response_text}")

    
    # Try to extract JSON content
    json_content = extract_json_content(response_text)
    print(f"json content is {json_content}")


    response_data = {
            "user_query": request.user_query,
            "response": {
                "response": json_content["response"],
                "created_charts": json_content["created_charts"]
            }
        }
    
    print("Response data:", response_data)
    
    # Update the project's chat_history in MongoDB
    projects_collection.update_one(
        {"project_id": request.project_id},
        {"$push": {"chat_history": response_data}}
    )

    return response_data


class ProjectModel(BaseModel):
    project_id: str
    project_name: str
    creation_time: datetime
    uploaded_files: List[dict] = []
    chat_history: List[dict] = []
    user_id: str

import json

# Define a function to get the project's charts directory
def get_project_charts_dir(project_name: str, project_id: str) -> str:
    return os.path.join(FILES_DIR, f"{project_name}_{project_id}", "created_charts")

# Mount static files for each project's charts directory
@app.get("/project/{project_id}")
async def get_project(project_id: str):
    project = projects_collection.find_one({"project_id": project_id})
    if not project:
        raise HTTPException(status_code=404, detail="Project not found.")

    # Convert ObjectId to string
    if "_id" in project:
        project["_id"] = str(project["_id"])

    # Ensure consistent response format in chat history
    if "chat_history" in project:
        for chat in project["chat_history"]:
            # If response is a string that looks like a JSON object
            if isinstance(chat.get("response"), str):
                try:
                    # Try to parse it as JSON
                    chat["response"] = json.loads(chat["response"].replace("'", "\""))
                except json.JSONDecodeError:
                    pass
            
            # Ensure created_charts exists and update paths to relative URLs
            if isinstance(chat.get("response"), dict):
                if "created_charts" not in chat["response"]:
                    chat["response"]["created_charts"] = []
                else:
                    # Convert absolute paths to relative URLs
                    charts = []
                    for chart_path in chat["response"]["created_charts"]:
                        filename = os.path.basename(chart_path)
                        charts.append(filename)
                    chat["response"]["created_charts"] = charts

    return project




from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, EmailStr
from pymongo import MongoClient
from bson import ObjectId
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta



# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT settings
SECRET_KEY = "my secret key is very hard to crack."
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_DAYS = 3

# Pydantic models
class UserSignup(BaseModel):
    full_name: str
    phone_number: str
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

# Utility functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()  # This will now include both 'sub' and 'user_id'
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# Signup endpoint
@app.post("/signup")
async def signup(user: UserSignup):
    # Check if user already exists
    existing_user = users_collection.find_one({"email": user.email})
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    # Create new user
    hashed_password = get_password_hash(user.password)
    new_user = {
        "full_name": user.full_name,
        "user_id" : str(uuid4()),
        "phone_number": user.phone_number,
        "email": user.email,
        "password": hashed_password,
        "is_verified": False
    }
    print(new_user)
    users_collection.insert_one(new_user)
    print("user registered successfully......")
    return {"message": "User registered successfully"}

# Login endpoint
@app.post("/login")
async def login(user: UserLogin):
    # Check if user exists
    existing_user = users_collection.find_one({"email": user.email})
    if not existing_user:
        raise HTTPException(status_code=400, detail="Invalid email or password")

    # Check if user is verified
    if not existing_user.get("is_verified"):
        raise HTTPException(status_code=400, detail="User is not verified")

    # Verify password
    if not verify_password(user.password, existing_user["password"]):
        raise HTTPException(status_code=400, detail="Invalid email or password")

    # Create JWT token - Modify this part to include user_id
    access_token = create_access_token(
        data={
            "sub": existing_user["email"],
            "user_id": existing_user["user_id"]  # Add user_id to token payload
        }
    )
    return {"access_token": access_token, "token_type": "bearer"}



# Pydantic Model for Response
class ProjectResponse(BaseModel):
    project_name: str
    project_id: str
    user_id: str
    creation_time: str  # Assuming stored as ISO format string



from fastapi.security import OAuth2PasswordBearer



# Define OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Function to decode JWT and get current user
def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_email = payload.get("sub")
        if not user_email:
            raise HTTPException(status_code=401, detail="Invalid token")

        # Fetch user from MongoDB based on email
        user = users_collection.find_one({"email": user_email}, {"_id": 0, "user_id": 1, "email": 1})
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        return user
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

from datetime import datetime

@app.get("/projects", response_model=List[ProjectResponse])
async def get_user_projects(current_user: dict = Depends(get_current_user)):
    user_id = current_user["user_id"]  # Extract user ID from authenticated user

    projects = list(projects_collection.find(
        {"user_id": user_id},
        {"_id": 0, "project_name": 1, "project_id": 1, "user_id": 1, "creation_time": 1}
    ))

    if not projects:
        raise HTTPException(status_code=404, detail="No projects found for this user.")

    # Convert creation_time to ISO format if it exists
    for project in projects:
        if "creation_time" in project and isinstance(project["creation_time"], datetime):
            project["creation_time"] = project["creation_time"].isoformat()

    return projects

# Mount the static files directory at the end of the file
charts_path = os.path.join(FILES_DIR)
app.mount("/static", StaticFiles(directory=charts_path), name="static")

@app.delete("/project/{project_id}")
async def delete_project(project_id: str, current_user: dict = Depends(get_current_user)):
    # Check if project exists and belongs to user
    project = projects_collection.find_one({
        "project_id": project_id,
        "user_id": current_user["user_id"]
    })
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    try:
        # Delete project folder and all its contents
        project_dir = os.path.join(FILES_DIR, f"{project['project_name']}_{project_id}")
        if os.path.exists(project_dir):
            shutil.rmtree(project_dir)
        
        # Delete project from MongoDB
        result = projects_collection.delete_one({
            "project_id": project_id,
            "user_id": current_user["user_id"]
        })
        
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Project not found")
            
        return {"message": "Project deleted successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete project: {str(e)}")

# Add this new endpoint
@app.put("/project/{project_id}/rename")
async def rename_project(
    project_id: str, 
    new_name: str = Form(...), 
    current_user: dict = Depends(get_current_user)
):
    # Check if project exists and belongs to user
    project = projects_collection.find_one({
        "project_id": project_id,
        "user_id": current_user["user_id"]
    })
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Check if user already has a project with the new name
    existing_project = projects_collection.find_one({
        "project_name": new_name,
        "user_id": current_user["user_id"],
        "project_id": {"$ne": project_id}  # Exclude current project
    })
    
    if existing_project:
        raise HTTPException(
            status_code=400, 
            detail="You already have a project with this name. Please choose a different name."
        )
    
    try:
        old_dir = os.path.join(FILES_DIR, f"{project['project_name']}_{project_id}")
        new_dir = os.path.join(FILES_DIR, f"{new_name}_{project_id}")
        
        # First update the MongoDB document
        result = projects_collection.update_one(
            {
                "project_id": project_id,
                "user_id": current_user["user_id"]
            },
            {"$set": {"project_name": new_name}}
        )
        
        if result.modified_count == 0:
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Then try to rename the directory if it exists
        if os.path.exists(old_dir):
            try:
                # Try to force close any open files (Windows specific)
                import gc
                gc.collect()  # Force garbage collection
                
                # If directory exists, try to rename it
                if os.path.exists(new_dir):
                    # If target directory already exists, merge contents
                    for item in os.listdir(old_dir):
                        old_item = os.path.join(old_dir, item)
                        new_item = os.path.join(new_dir, item)
                        if os.path.isdir(old_item):
                            shutil.copytree(old_item, new_item, dirs_exist_ok=True)
                        else:
                            shutil.copy2(old_item, new_item)
                    shutil.rmtree(old_dir)
                else:
                    # If target directory doesn't exist, try to rename
                    shutil.move(old_dir, new_dir)
                
            except Exception as e:
                # If directory rename fails, log it but don't fail the request
                print(f"Warning: Failed to rename directory: {str(e)}")
                # The MongoDB update was successful, so we'll return success
                # but include a warning in the response
                return {
                    "message": "Project renamed partially",
                    "new_name": new_name,
                    "warning": "Database updated but directory rename failed. Please close any open files and try again."
                }
            
        return {"message": "Project renamed successfully", "new_name": new_name}
        
    except Exception as e:
        # If MongoDB update failed, revert everything
        if result and result.modified_count > 0:
            try:
                # Revert MongoDB change
                projects_collection.update_one(
                    {"project_id": project_id},
                    {"$set": {"project_name": project["project_name"]}}
                )
            except:
                pass
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to rename project: {str(e)}"
        )