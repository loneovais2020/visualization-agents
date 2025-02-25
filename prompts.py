def file_summary_generation_prompt(content):
    prompt = f"""

Please analyze the following document and provide a concise summary between 50 to 100 words. The summary should encapsulate the main themes and key points, clearly stating the document's purpose and subject matter. Ensure that the summary is informative and accurately reflects the content of the document. Make sure there is no additional text or commentary which is unrelated to file content.\n\n

Document Content:
{content}

"""
    
    return prompt



def source_to_select_prompt(file_summary_dict, user_query):
    return f"""

You are an AI assistant tasked with determining the best approach to answer the user's query. Based on the provided file summaries and the user's question, decide which resources to utilize. Your response should be in valid JSON format with the following keys:

- `files_to_use`: A list of filenames that are relevant to answering the query. If none are applicable, return an empty list.
- `use_internet`: A boolean value indicating whether accessing the internet is necessary (`true`) or not (`false`).
- `self_knowledge`: A boolean value indicating whether your existing knowledge is sufficient to answer the query (`true`) or not (`false`).

Note: At any given time, you should choose only one of the following options:
- Use specific files (`files_to_use` is non-empty, `use_internet` and `self_knowledge` are `false`).
- Access the internet (`use_internet` is `true`, `files_to_use` is empty, and `self_knowledge` is `false`).
- Rely on your own knowledge (`self_knowledge` is `true`, `files_to_use` is empty, and `use_internet` is `false`).

Ensure that the JSON response adheres to this structure and does not include any additional commentary or formatting.

**File Summaries**:
{file_summary_dict}

**User Query**:
{user_query}

"""


def answer_user_query_prompt(file_path, user_query):
    return f"""

You are an AI assistant equipped with code execution capabilities. Your task is to analyze the provided dataset and address the user's query by generating and executing Python code.  Ensure that your response includes the necessary code and its output, such as graphs or analysis results.



**User Query**:

{user_query}

**Instructions**:
1. **Data Loading**: Read the dataset using the provided filepath into an appropriate data structure (e.g., a Pandas DataFrame).
2. **Data Analysis/Visualization**: Based on the user's query, perform the necessary data analysis or create visualizations. Utilize libraries such as Pandas, Matplotlib, Seaborn, or Plotly as needed.
3. **Code Execution**: Execute the generated Python code to produce the desired output.
4. **Output Presentation**: Display the results, including any graphs or analysis summaries, directly below the code.

**Example**:

If the user query is: "Please provide a bar chart of the top 5 categories by sales volume," your response should include:

- Python code that reads the dataset, processes the data to find the top 5 categories by sales volume, and generates a bar chart.
- The executed output, displaying the bar chart.

**Note**: Ensure that the code is efficient and handles potential errors, such as missing values or incorrect data types.

**File path is**:

{file_path}
"""