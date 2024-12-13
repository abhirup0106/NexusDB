#import time
from flask import Flask, request, render_template, jsonify, redirect, url_for, flash, session
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI  
from langchain_experimental.sql import SQLDatabaseChain  
from langchain.utilities import SQLDatabase  
from langchain.prompts import PromptTemplate  
from langchain.chains import LLMChain 
import mysql.connector  
import pymysql  
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings  
import os
from dotenv import load_dotenv  
from hashlib import sha256  
import re
from flask_cors import CORS 
from mysql.connector import Error  
import sqlite3
from langchain import OpenAI
import PyPDF2
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import pandas as pd
from PIL import Image
import pytesseract
import docx
from docx import Document
from functools import lru_cache
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError



SQLDatabaseChain.model_rebuild()

CONTEXTUAL_SYNONYMS = {
    "type": "type",
    "category": "type",  # Map 'category' to 'type' in the query
    "item": "type",      # Optionally handle 'item' or similar terms
}
# Initialize Flask app
app = Flask(__name__)




app.secret_key = 'your_secret_key'  # Required for session management

# Dummy user credentials for verification
USER_CREDENTIALS = {
    'test@example.com': 'password123'
}

USER_CREDENTIALS_Child = {
    'child@example.com': 'pass12345'
}
# Google Gemini AI setup
api_key = "AIzaSyDWW5LQbReUDSqJbQnS0XXouWD_iI6B-lI"
genai.configure(api_key=api_key)
model = "models/gemini-pro"

try:
    llm = ChatGoogleGenerativeAI(api_key=api_key, model=model)
except Exception as e:
    print("Error initializing Google Generative AI:", e)
    raise

ADMIN_CREDENTIALS = {"admin@example.com": "password123"}

# Mock store list (database names)





def connect_to_store_db(store_id):
    """Create a dynamic connection to the selected store database."""
    try:
        # Fetch database configuration for the selected store ID
        if store_id not in DATABASES:
            raise ValueError(f"Invalid store ID: {store_id}")
        
        db_info = DATABASES[store_id]
        db_name = db_info["db_name"]
        connection_uri = (
            f"mysql+pymysql://{db_info['db_user']}:"
            f"{db_info['db_password']}@{db_info['db_host']}/{db_name}"
        )
        
        # Create and return the SQLAlchemy engine
        engine = create_engine(connection_uri)
        return engine
    except ValueError as ve:
        raise ValueError(str(ve))
    except Exception as e:
        raise RuntimeError(f"Error connecting to database '{db_name}': {e}")

    
    
@app.route("/open-store", methods=["POST"])
def open_store():
    """Handle store selection and redirect to the store page."""
    if not session.get("logged_in"):
        return redirect("/")
    
    # Get the selected store from the form
    selected_store = request.form["storeName"]
    session["store"] = selected_store  # Save selected store in session

    return redirect("/store")


@app.route("/store", methods=["GET"])
def store_index():
    """Display the data from the selected store."""
    if not session.get("logged_in"):
        return redirect("/")
    
    # Get the selected store from the session
    store_id = session.get("store")
    if not store_id:
        return "No store selected."

    try:
        # Validate the selected store
        if store_id not in DATABASES:
            return f"Invalid store selected: {store_id}.", 400
        
        # Fetch database details for the selected store
        db_info = DATABASES[store_id]

        # Create the database connection using SQLAlchemy
        engine = create_engine(
            f"mysql+mysqlconnector://{db_info['db_user']}:{db_info['db_password']}@{db_info['db_host']}/{db_info['db_name']}"
        )

        # Dynamically determine table name
        table_name = f"store_{store_id[-1]}"  # e.g., store1 -> store_1, store2 -> store_2

        # Query to fetch all rows from the dynamically selected table
        query = f"SELECT * FROM {table_name};"
        with engine.connect() as conn:
            data = pd.read_sql(query, conn)

        # Convert the data to HTML for rendering
        data_html = data.to_html(classes="table table-striped", index=False)

        # Render the HTML template with the table data
        return render_template("store.html", store=store_id, data_html=data_html)

    except SQLAlchemyError as e:
        return f"Failed to connect to {store_id}. Error: {str(e)}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"



@app.route('/')
def home():
    return render_template('revolve.html')

@app.route('/parchi',)
def parchi():
    return render_template('parchi.html')



@app.route('/index2')
def index2():
    return render_template('index2.html')

@app.route('/index', methods=['GET'])
def index():
    # if not session.get('logged_in'):  # Check if the user is logged in
    #     return redirect(url_for('parchi'))  # Redirect to login if not authenticated

    # Get the selected store from the query parameter
   
        # Render the template and pass the store and table information
    return render_template('index.html')


        # Handle database connection errors
       


   


# Render the protected page

@app.route('/logout')
def logout():
    session.pop('logged_in', None)  # Remove the logged-in flag from the session
    return redirect(url_for('parchi'))

@app.route('/check')
def check():
    session.pop('logged_in', None)  # Remove the logged-in flag from the session
    return redirect(url_for('child_login'))





@app.route('/index1')
def index1():
    return render_template('index1.html')

# @app.route("/open-store", methods=["POST"])
# def open_store():
#     # Access the selected store from the form
#     selected_store = request.form.get("storeName")
    
#     # Check the selected store and assign the corresponding database name
#     if selected_store == "fashionstore1":
#         db_name = "fashionstore1"
#     elif selected_store == "fashionstore2":
#         db_name = "fashionstore2"
#     else:
#         return "Invalid store selection. Please go back and try again."
    
#     # Save the database name in the session for use in subsequent routes
#     session["store"] = db_name
#     return redirect("/index")


DATABASES = {
    "store1": {
        "db_user": "root",
        "db_password": "userrt",
        "db_host": "localhost",
        "store1_db_name": "fashionstore1"
    },
    "store2": {
        "db_user": "root",
        "db_password": "userrt",
        "db_host": "localhost",
        "store2_db_name": "fashionstore2"
    },
    "store3":{
        "db_user": "root",
        "db_password": "userrt",
        "db_host": "localhost",
        "store3_db_name": "fashionstore3"
    }
    # Add more stores as needed
}

@app.route('/run_query', methods=['POST'])
def run_query():
    try:
        # Get the selected store ID from the request
        data = request.json
        store_id = data.get("store_id")

        if not store_id or store_id not in DATABASES:
            return jsonify({"message": "Invalid store selected"}), 400

        # Fetch the database connection details dynamically
        db_info = DATABASES[store_id]

        # Extract the database name based on the store_id
        db_name_key = f"{store_id}_db_name"
        db_name = db_info.get(db_name_key)

        if not db_name:
            return jsonify({"message": "Database name not configured correctly"}), 400

        # Connect to the selected database
        conn = mysql.connector.connect(
            user=db_info["db_user"],
            password=db_info["db_password"],
            host=db_info["db_host"],
            database=db_name
        )

        # Execute a query (modify as needed)
        cursor = conn.cursor()

        # Example query: Fetch data from products table (update table name accordingly)
        query = f"SELECT * FROM store_{store_id[-1]} LIMIT 10;"        
        cursor.execute(query)
        result = cursor.fetchall()

        # Close connection
        cursor.close()
        conn.close()

        # Return the results
        return jsonify({"message": "Query executed successfully", "result": result}), 200

    except mysql.connector.Error as err:
        if err.errno == 1146:  # Table not found error
            return jsonify({"message": f"Table does not exist in the selected database: {err}"}), 400
        else:
            return jsonify({"message": f"Database error: {err}"}), 500
    except Exception as e:
        return jsonify({"message": f"An unexpected error occurred: {e}"}), 500




# @app.route('/select_store', methods=['POST'])
# def select_store():
#     try:
#         # Ensure request has JSON content
#         if not request.is_json:
#             return jsonify({"success": False, "message": "Invalid Content-Type. Expected application/json."}), 400

#         # Parse JSON data
#         data = request.get_json()
#         store_id = data.get('store_id', '').strip()

#         # Validate store_id
#         if store_id not in DATABASES:
#             return jsonify({"success": False, "message": f"Invalid store ID: {store_id}"}), 400

#         # Store the selected store_id in a session
#         session['store_id'] = store_id
#         print(f"Store {store_id} selected successfully!")
#         return jsonify({"success": True, "message": f"Store {store_id} selected successfully!"}), 200
#     except Exception as e:
#         print("Error selecting store:", e)
#         return jsonify({"success": False, "message": "Unexpected error occurred"}), 500




# Database connection setup


def get_database_connection(store_id):
    """Get a database connection for the specified store."""
    try:
        # Retrieve the database configuration for the given store ID
        if store_id not in DATABASES:
            raise ValueError(f"Invalid store ID: {store_id}")

        db_info = DATABASES[store_id]
        conn = mysql.connector.connect(
            user=db_info["db_user"],
            password=db_info["db_password"],
            host=db_info["db_host"],
            database=db_info[f"{store_id}_db_name"]
        )
        return conn
    except Exception as e:
        print(f"Error connecting to database for store '{store_id}':", e)
        raise
cors = CORS()
cors.init_app(app)


#OLD
# Database connection setup
db_user1 = "root"
db_password1 = "userrt"
db_host1 = "localhost"
db_name1 = "atliq_tshirts"

# Initialize the database
#db_connection_string = f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}"
db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user1}:{db_password1}@{db_host1}/{db_name1}",sample_rows_in_table_info=3)
db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)

# Vectorization setup
embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key="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",
        model_name="sentence-transformers/all-MiniLM-l6-v2"
    )

sql_prompt_template1 = """
You are an expert SQL query generator. Below is the schema for the t_shirts table:

t_shirts(price, color,size,stock_quantity, brand)

Given a natural language input, generate a correct SQL query without any prefixes or comments, only valid SQL.

Question: {question1}

SQL Query:
"""
sql_prompt = PromptTemplate(
    input_variables=["question1"],
    template=sql_prompt_template1
)
sql_chain1 = LLMChain(llm=llm, prompt=sql_prompt)


FUZZY_THRESHOLD = 85



def normalize_input(text):
    text = text.lower().strip()
    text = re.sub(r'[\s\-]+', '', text)  # Remove spaces and hyphens
    return text


def fuzzy_match_input(input_text, valid_options):
    normalized_input = normalize_input(input_text)
    match, score = process.extractOne(normalized_input, valid_options, scorer=fuzz.ratio)
    return match if score >= FUZZY_THRESHOLD else None


# SQL prompt template with better normalization
# sql_prompt_template = """
# You are an expert SQL query generator. Below is the schema for the tshirt_priyam table:

# tshirt_priyam(brand, size, color,  fabric , style , pattern ,sleeve , gender ,occasion , stock_quantity)

# Generate consistent SQL queries for given questions. Ensure no extraneous comments or formatting issues in the output.

# Question: {question}

# SQL Query:
# """
# sql_prompt = PromptTemplate(
#     input_variables=["question"],
#     template=sql_prompt_template
# )

table_name = 'nexusdb'

# def fetch_tables_mysql(conn,table_name):
#     """
#     Fetch table names from the specified MySQL database.
#     """
#     try:
#         # Connect to the database
#         conn = mysql.connector.connect(
#             user=db_user,
#             password=db_password,
#             host=db_host,
#             database=db_name
#         )
#         cursor = conn.cursor()
#         query = """
#     SELECT COLUMN_NAME
#     FROM information_schema.columns
#     WHERE table_name = %s AND table_schema = DATABASE()
#     ORDER BY ORDINAL_POSITION;
#     """
#         cursor.execute(query)
#         tables = [row[0] for row in cursor.fetchall()]
        
#         # Close connection
#         cursor.close()
#         conn.close()
        
#         return tables
#     except mysql.connector.Error as e:
#         print(f"Error fetching tables: {e}")
#         return []




# Function to fetch the schema of the database
def get_database_schema(conn):
    """
    Fetch the database schema as a dictionary mapping table names to column names.
    """
    schema = {}
    cursor = conn.cursor()
    cursor.execute("SHOW TABLES")
    tables = [row[0] for row in cursor.fetchall()]

    for table in tables:
        cursor.execute(f"DESCRIBE {table}")
        columns = [row[0] for row in cursor.fetchall()]
        schema[table] = columns

    cursor.close()
    return schema

# Function to fetch distinct values for a specific column
def fetch_column_values(conn, table_name, column_name):
    try:
        cursor = conn.cursor()
        query = f"SELECT DISTINCT `{column_name}` FROM `{table_name}`"
        cursor.execute(query)
        values = [row[0] for row in cursor.fetchall()]
        cursor.close()
        return values
    except mysql.connector.Error as e:
        print(f"Error fetching values for column {column_name}: {e}")
        return []

# Generate SQL prompt template with column values
def generate_sql_prompt_template_with_values(conn, schema):
    schema_description = ""
    for table, columns in schema.items():
        schema_description += f"{table}("
        column_descriptions = []
        for column in columns:
            values = fetch_column_values(conn, table, column)
            column_descriptions.append(
                f"{column} ({', '.join(map(str, values))})"
            )
        schema_description += ", ".join(column_descriptions) + ")\n"
    
    template = f"""
You are an expert SQL query generator. Below is the schema for the database, including column names and possible values:

{schema_description}
Use this schema to generate valid SQL queries. Only use the provided column names and their values. Avoid assuming or generalizing terms.

Rules:
1. Always use the `Type` column to describe product types (e.g., Shirts, Sarees).



Question: {{question}}

SQL Query:
"""

    return template

# Preprocess user input to avoid unnecessary transformations
def preprocess_input(question):
    # Remove leading/trailing spaces
    question = question.strip()

    # Remove multiple question marks and reduce them to a single one
    question = re.sub(r'\?{2,}', '?', question)

    # Remove trailing numbers followed by a question mark (optional, if needed)
    question = re.sub(r"\s?\d+\?$", "", question)

    # Replace multiple spaces with a single space
    question = re.sub(r'\s+', ' ', question)

    return question



# Validate the AI-generated SQL query
def validate_sql_query(schema, query):
    for table, columns in schema.items():
        for column in columns:
            if column in query and table not in query:
                raise ValueError(f"Invalid reference to column '{column}' without table '{table}'")
    return True

# Simple cache for query responses
query_cache = {}

def get_cached_response(question):
    hashed_question = sha256(question.encode()).hexdigest()
    return query_cache.get(hashed_question)

def cache_response(question, response):
    hashed_question = sha256(question.encode()).hexdigest()
    query_cache[hashed_question] = response
    
    
    

# Run prediction for user input
# def run_prediction(question, store_id):
#     question = preprocess_input(question)

#     # Check for cached response
#     cached_response = get_cached_response(question)
#     if cached_response:
#         print("Using cached response")
#         return cached_response

#     # Initialize variables
#     sql_query = None

#     try:
#         # Generate SQL query using the custom prompt
#         ai_response = sql_chain.invoke({"question": question})

#         # Extract text if 'text' attribute is available; otherwise, convert directly to string
#         sql_query = ai_response.text if hasattr(ai_response, 'text') else str(ai_response)

#         # Clean the generated SQL query
#         sql_query = re.sub(r"(additional_kwargs=.*|response_metadata=.*|id=.*|usage_metadata=.*)", '', sql_query)
#         sql_query = re.sub(r'^content="```', '', sql_query)
#         sql_query = re.sub(r'(```"|\"$)', '', sql_query)
#         sql_query = re.sub(r'(\\n|\\r|\\t)', ' ', sql_query)
#         sql_query = re.sub(r'\s+', ' ', sql_query).strip()
#         sql_query = sql_query.lstrip('sql').lstrip()

#         if "content=" in sql_query:
#             sql_query = sql_query.split("content=")[-1].strip("'").strip()

#         if "sql" in sql_query:
#             sql_query = sql_query.split("sql")[1].strip("'").strip()

#         sql_query = re.sub(r'`+$', '', sql_query)

#         # Ensure valid SQL query
#         validate_sql_query(schema, sql_query)

#         # Get the database connection details for the store
#         if store_id not in DATABASES:
#             raise ValueError(f"Invalid store ID: {store_id}")
#         db_info = DATABASES[store_id]

#         # Dynamically get the database name
#         db_name_key = f"{store_id}_db_name"
#         db_name = db_info.get(db_name_key)
#         if not db_name:
#             raise ValueError(f"Database name not found for store ID: {store_id}")

#         # Connect to the database
#         conn = mysql.connector.connect(
#             user=db_info["db_user"],
#             password=db_info["db_password"],
#             host=db_info["db_host"],
#             database=db_name
#         )

#         # Execute the SQL query
#         cursor = conn.cursor()
#         cursor.execute(sql_query)
#         result = cursor.fetchall()

#         # Close connection
#         cursor.close()
#         conn.close()

#         # Prepare response
#         response = {
#             "sql_query": sql_query,
#             "result": result
#         }

#         # Cache and return the response
#         cache_response(question, response)
#         return response

#     except mysql.connector.Error as err:
#         print("MySQL Error:", err)
#         return {"error": str(err), "sql_query": sql_query if sql_query else "N/A"}
#     except ValueError as ve:
#         print("Value Error:", ve)
#         return {"error": "Value error: " + str(ve), "sql_query": sql_query if sql_query else "N/A"}
#     except Exception as e:
#         print("Unexpected Error:", e)
#         return {"error": "Unexpected error: " + str(e), "sql_query": sql_query if sql_query else "N/A"}


# # Main execution
# try:
#     # Define a question and store ID
#     question = "What is the stock for menswear printed synthetic T-shirts?"
  
    
#     store_id = "store1"
 
#  # Replace with the actual store ID as needed

#     # Fetch the database connection details dynamically
#     if store_id not in DATABASES:
#         raise ValueError(f"Invalid store ID: {store_id}")
#     db_info = DATABASES[store_id]

#     # Dynamically get the database name
#     db_name_key = f"{store_id}_db_name"
#     db_name = db_info.get(db_name_key)
#     if not db_name:
#         raise ValueError(f"Database name not found for store ID: {store_id}")

#     # Connect to the database
#     conn = mysql.connector.connect(
#         user=db_info["db_user"],
#         password=db_info["db_password"],
#         host=db_info["db_host"],
#         database=db_name
#     )

#     # Fetch schema and generate prompt template
#     schema = get_database_schema(conn)
#     dynamic_sql_prompt_template = generate_sql_prompt_template_with_values(conn, schema)

#     # Initialize the SQL prompt
#     sql_prompt = PromptTemplate(
#         input_variables=["question"],
#         template=dynamic_sql_prompt_template
#     )

#     # Create the SQL generation pipeline
#     sql_chain = sql_prompt | llm

#     # Run the prediction
#     response = run_prediction(question, store_id)
#     print(response)

# except Exception as e:
#     print("Error initializing SQLDatabase or prompt template:", e)
#     raise

# @app.route('/run_prediction', methods=['POST'])
# def run_prediction_api():
#     try:
#         # Get the request JSON data
#         data = request.json
#         question = data.get("question")
#         store_id = data.get("store_id")

#         # Validate store_id
#         if not store_id or store_id not in DATABASES:
#             return jsonify({"message": "Invalid store selected"}), 400

#         # Fetch the database connection details for the store
#         db_info = DATABASES[store_id]

#         # Dynamically get the database name
#         db_name_key = f"{store_id}_db_name"
#         db_name = db_info.get(db_name_key)
#         if not db_name:
#             return jsonify({"message": "Database name not configured correctly"}), 400

#         # Connect to the database to get the schema
#         conn = mysql.connector.connect(
#             user=db_info["db_user"],
#             password=db_info["db_password"],
#             host=db_info["db_host"],
#             database=db_name
#         )

#         # Fetch schema and generate prompt template
#         schema = get_database_schema(conn)
#         dynamic_sql_prompt_template = generate_sql_prompt_template_with_values(conn, schema)

#         # Initialize the SQL prompt
#         sql_prompt = PromptTemplate(
#             input_variables=["question"],
#             template=dynamic_sql_prompt_template
#         )

#         # Create the SQL generation pipeline
#         sql_chain = sql_prompt | llm

#         # Run the prediction using the dynamic `store_id`
#         response = run_prediction(question, store_id)

#         return jsonify({"message": "Prediction successful", "data": response}), 200

#     except mysql.connector.Error as err:
#         return jsonify({"message": f"Database error: {err}"}), 500
#     except Exception as e:
#         return jsonify({"message": f"An unexpected error occurred: {e}"}), 500

def run_prediction(question, store_id):
    question = preprocess_input(question)

    # Check for cached response
    cached_response = get_cached_response(question)
    if cached_response:
        print(f"Using cached response: {cached_response}")
        return cached_response

    # Initialize variables
    sql_query = None

    try:
        # Get the database connection details for the store
        if store_id not in DATABASES:
            raise ValueError(f"Invalid store ID: {store_id}")
        db_info = DATABASES[store_id]

        # Dynamically get the database name
        db_name_key = f"{store_id}_db_name"
        db_name = db_info.get(db_name_key)
        if not db_name:
            raise ValueError(f"Database name not found for store ID: {store_id}")

        # Connect to the database to fetch schema
        conn = mysql.connector.connect(
            user=db_info["db_user"],
            password=db_info["db_password"],
            host=db_info["db_host"],
            database=db_name
        )

        # Fetch the schema dynamically
        schema = get_database_schema(conn)

        # Generate the SQL prompt template dynamically
        dynamic_sql_prompt_template = generate_sql_prompt_template_with_values(conn, schema)

        # Initialize the SQL prompt
        sql_prompt = PromptTemplate(
            input_variables=["question"],
            template=dynamic_sql_prompt_template
        )

        # Create the SQL generation pipeline
        sql_chain = sql_prompt | llm

        # Generate SQL query using the custom prompt
        ai_response = sql_chain.invoke({"question": question})

        # Extract and clean the SQL query
        sql_query = ai_response.text if hasattr(ai_response, 'text') else str(ai_response)
        sql_query = re.sub(r"(additional_kwargs=.*|response_metadata=.*|id=.*|usage_metadata=.*)", '', sql_query)
        sql_query = re.sub(r'^content="```', '', sql_query)
        sql_query = re.sub(r'(```"|\"$)', '', sql_query)
        sql_query = re.sub(r'(\\n|\\r|\\t)', ' ', sql_query)
        sql_query = re.sub(r'\s+', ' ', sql_query).strip()
        sql_query = sql_query.lstrip('sql').lstrip()
        sql_query = sql_query.replace("\\'", "'")
        sql_query = sql_query.replace("\\\\", "\\")

        if "content=" in sql_query:
            sql_query = sql_query.split("content=")[-1].strip("'").strip()

        if "sql" in sql_query:
            sql_query = sql_query.split("sql")[1].strip("'").strip()

        sql_query = re.sub(r'`+$', '', sql_query)

        # Ensure valid SQL query
        validate_sql_query(schema, sql_query)

        # Dynamically determine the table name based on store_id
        table_name = f"store_{store_id[-1]}"
        
        # Replace any placeholder dynamically in the SQL query
        sql_query = re.sub("store_2" ,table_name, sql_query)

        # Execute the SQL query
        cursor = conn.cursor()
        cursor.execute(sql_query)
        result = cursor.fetchall()

        # Close connection
        cursor.close()
        conn.close()

        # Prepare response
        response = {
            "sql_query": sql_query,
            "result": result
        }

        # Cache and return the response
        #cache_response(question, response)
        
        return (result)
       

    except mysql.connector.Error as err:
        print("MySQL Error:", err)
        return {"error": str(err), "sql_query": sql_query if sql_query else "N/A"}
    except ValueError as ve:
        print("Value Error:", ve)
        return {"error": "Value error: " + str(ve), "sql_query": sql_query if sql_query else "N/A"}
    except Exception as e:
        print("Unexpected Error:", e)
        return {"error": "Unexpected error: " + str(e), "sql_query": sql_query if sql_query else "N/A"}




# ai suggestion

def preprocess_input(question1):
    # Add any preprocessing steps for the question if needed
    return question1.strip()

def run_prediction1(question1):
    try:
        # Generate SQL query using the custom prompt
        sql_query = sql_chain1.run(question1)

        # Log the raw SQL query to understand the formatting issues
        print("Raw SQL Query Generated:", sql_query)
        
        # Clean up the SQL query by removing unwanted formatting
        sql_query = sql_query.replace('sql', '').replace('```', '').strip()
        sql_query = sql_query.replace("\n", " ").strip()

        print("Cleaned SQL Query:", sql_query)
        
        # Directly execute the SQL query using MySQL connector
        conn = mysql.connector.connect(
            user=db_user1,
            password=db_password1,
            host=db_host1,
            database=db_name1
        )
        cursor = conn.cursor()
        cursor.execute(sql_query)
        result = cursor.fetchall()
        
        # Print result for debugging
        print("Query Result:", result)
        
        # Close connection
        cursor.close()
        conn.close()

        # Convert result to JSON-compatible format
        response = {"result": result}
        return response
    except mysql.connector.Error as err:
        print("MySQL Error:", err)
        return {"error": str(err)}
    except Exception as e:
        print("Error:", e)
        return {"error": str(e)}



@app.route('/get_dropdown_values', methods=['POST'])
def get_dropdown_values():
    try:
        # Get JSON data from the request
        data = request.json or {}
        store_id = data.get("store_id")
        column_names = data.get('column_names', [])  # Default to empty list if not provided

        # Validate the store_id and get database info
        if not store_id or store_id not in DATABASES:
            return jsonify({"success": False, "message": "Invalid store selected"}), 400

        db_info = DATABASES[store_id]
        db_name_key = f"{store_id}_db_name"
        db_name = db_info.get(db_name_key)

        if not db_name:
            return jsonify({"success": False, "message": "Database name not configured correctly"}), 400

        # Connect to the selected database
        conn = mysql.connector.connect(
            user=db_info["db_user"],
            password=db_info["db_password"],
            host=db_info["db_host"],
            database=db_name
        )
        cursor = conn.cursor()

        # Define the table name based on the store_id
        table_name = f"store_{store_id[-1]}"

        # Dynamically fetch all column names from the table
        cursor.execute(f"SHOW COLUMNS FROM {table_name};")
        db_columns = [row[0].strip() for row in cursor.fetchall()]

        # Log the fetched columns for debugging
        print(f"Fetched columns from table '{table_name}':", db_columns)

        # Exclude 'Stock_Quantity' and any columns related to 'Stock'
        excluded_columns = ['stock_quantity', 'stock']
        db_columns = [col for col in db_columns if col.lower() not in excluded_columns]

        # Log the filtered columns for debugging
        print("Filtered columns (excludes 'Stock_Quantity' and 'Stock'):", db_columns)

        # If no column names provided, use all columns except excluded ones
        if not column_names:
            column_names = db_columns

        # Validate requested column names
        invalid_columns = [col for col in column_names if col not in db_columns]
        if invalid_columns:
            return jsonify({
                "success": False,
                "message": f"Invalid column names: {', '.join(invalid_columns)}"
            }), 400

        # Check if 'BrandName' column is selected; if not, return an error
        if 'BrandName' not in column_names:
            return jsonify({
                "success": False,
                "message": "'BrandName' column is mandatory. Please include 'BrandName' in the selected columns."
            }), 400

        # Fetch distinct values for each requested column
        result_data = {}
        for column_name in column_names:
            if column_name.lower() in excluded_columns:
                continue
            sql_query = f"SELECT DISTINCT `{column_name}` FROM `{table_name}`;"
            cursor.execute(sql_query)
            results = cursor.fetchall()
            result_data[column_name] = [row[0] for row in results]

        # Close the connection
        cursor.close()
        conn.close()

        # Log the result data for debugging
        print("Result Data Sent to Frontend:", result_data)

        # Return the dropdown values
        return jsonify({"success": True, "values": result_data}), 200

    except mysql.connector.Error as err:
        print("MySQL Error:", err)
        return jsonify({"success": False, "message": str(err)}), 500
    except Exception as e:
        print("Error:", e)
        return jsonify({"success": False, "message": "Unexpected error: " + str(e)}), 500




@app.route('/get_stock_availability', methods=['POST'])
def get_stock_availability():
    try:
        # Get the JSON data from the request
        data = request.json or {}
        store_id = data.get('store_id')
        selected_values = data.get('selected_values', {})

        if not store_id:
            return jsonify({
                "success": False,
                "message": "Store ID is missing from the request."
            }), 400

        print("Received Store ID:", store_id)
        print("Received Selected Filters:", selected_values)

        # Validate the store_id
        if store_id not in DATABASES:
            return jsonify({
                "success": False,
                "message": f"Invalid store ID: {store_id}"
            }), 400

        # Database and query logic remains unchanged...

        # Get database info for the store
        db_info = DATABASES[store_id]
        db_name = db_info.get(f"{store_id}_db_name")

        if not db_name:
            return jsonify({
                "success": False,
                "message": "Database name not configured for the store."
            }), 400

        # Connect to the database
        conn = mysql.connector.connect(
            user=db_info["db_user"],
            password=db_info["db_password"],
            host=db_info["db_host"],
            database=db_name
        )
        cursor = conn.cursor()

        # Determine the table name dynamically
        table_name = f"store_{store_id[-1]}" 

        # Build the query dynamically based on selected filters
        filters = []
        for column, value in selected_values.items():
            if value.strip():  # Only add filters for non-empty values
                filters.append(f"{column} = %s")

        where_clause = " AND ".join(filters) if filters else "1=1"
        query = f"SELECT ID, BrandName, Stock FROM {table_name} WHERE {where_clause};"
        print("Constructed Query:", query)

        # Execute the query with parameterized values to prevent SQL injection
        query_values = [value.strip() for value in selected_values.values() if value.strip()]
        cursor.execute(query, query_values)
        results = cursor.fetchall()

        # Close the connection
        cursor.close()
        conn.close()

        print("Query Results:", results)

        # Parse the results to find available combinations
        available_combinations = [
            {"id": row[0], "brand_name": row[1], "stock_quantity": row[2]}
            for row in results if row[2] > 0  # Filter rows with stock > 0
        ]

        # Return the response
        return jsonify({
            "success": True,
            "available_combinations": available_combinations
        }), 200

    except mysql.connector.Error as err:
        print("MySQL Error:", err)
        return jsonify({"success": False, "message": f"Database error: {err}"}), 500
    except Exception as e:
        print("Unexpected Error:", e)
        return jsonify({"success": False, "message": f"Unexpected error: {e}"}), 500

    

#its mine 
# @app.route('/get_dropdown_values', methods=['POST'])
# def get_dropdown_values():
#     try:
#         # Expect filters in the POST request
#         data = request.json
#         filters = data.get('filters')  # Expecting a dictionary of filters, e.g., {"company": "Nike"}

#         if not filters:
#             return jsonify({"success": False, "message": "Filters are required."}), 400

#         # Normalize filter keys to lowercase for case-insensitive matching
#         normalized_filters = {key.lower(): value for key, value in filters.items()}

#         # Validate that the mandatory "company" field is present
#         if "company" not in normalized_filters:
#             return jsonify({
#                 "success": False,
#                 "message": "The 'company' filter is required."
#             }), 400

#         # Connect to the database
#         conn = mysql.connector.connect(
#             user=db_user,
#             password=db_password,
#             host=db_host,
#             database=db_name
#         )
#         cursor = conn.cursor()

#         # Fetch column names from the database
#         cursor.execute("SHOW COLUMNS FROM fashionhub_products;")
#         db_columns = [row[0].lower() for row in cursor.fetchall()]  # Extract column names (lowercased)

#         # Only include valid filters (present in db_columns)
#         valid_filters = {key: value for key, value in normalized_filters.items() if key in db_columns}

#         # Ensure "company" is included in the filters
#         if "company" not in valid_filters:
#             return jsonify({
#                 "success": False,
#                 "message": "The 'company' filter is required and must match a valid column."
#             }), 400

#         # Build WHERE clause dynamically based on filters
#         filter_clauses = [f"{key} = %s" for key in valid_filters.keys()]
#         filter_values = list(valid_filters.values())
#         filter_conditions = f"WHERE {' AND '.join(filter_clauses)}"

#         # Query to fetch the stock quantity based on filters
#         sql_query = f"SELECT SUM(stock_quantity) FROM fashionhub_products {filter_conditions};"
#         cursor.execute(sql_query, filter_values)
#         stock_quantity = cursor.fetchone()[0]

#         # Close the connection
#         cursor.close()
#         conn.close()

#         # Return the stock quantity
#         return jsonify({
#             "success": True,
#             "stock_quantity": stock_quantity if stock_quantity else 0
#         }), 200

#     except mysql.connector.Error as err:
#         print("MySQL Error:", err)
#         return jsonify({"success": False, "message": str(err)}), 500
#     except Exception as e:
#         print("Error:", e)
#         return jsonify({"success": False, "message": "Unexpected error: " + str(e)}), 500
    
    
    



# @app.route('/process-pdf', methods=['POST'])
# def process_pdf():
#     try:
#         pdf_file = request.files['pdf']
#         reader = PyPDF2.PdfReader(pdf_file)
        
#         # Extract text from all pages of the PDF
#         text = "".join(page.extract_text() for page in reader.pages)
        
#         # Extract questions from the text (simple example using split, customize as needed)
#         questions = extract_questions_from_text(text)
#         if not questions:
#             return jsonify({"success": False, "message": "No questions found in the PDF"}), 400

#         # Query the database for each question
#         answers = []
#         for question in questions:
#             sql_query = generate_sql_query(question)  # Convert question to SQL query
#             if not sql_query:
#                 answers.append({"question": question, "answer": "Could not generate query"})
#                 continue

#             # Execute the query and get results
#             results = query_database(sql_query)
#             if results:
#                 answers.append({"question": question, "answer": results})
#             else:
#                 answers.append({"question": question, "answer": "No results found"})

#         return jsonify({"success": True, "answers": answers})

#     except mysql.connector.Error as err:
#         print("MySQL Error:", err)
#         return jsonify({"success": False, "message": str(err)}), 500
#     except Exception as e:
#         print("Error:", e)
#         return jsonify({"success": False, "message": "Unexpected error: " + str(e)}), 500

# def extract_questions_from_text(text):
#     """
#     Extracts questions from the given text. This is a placeholder; update based on the PDF format.
#     """
#     # Example: Split text by newline and return lines containing "?" as questions
#     return [line.strip() for line in text.split("\n") if "?" in line]

# def generate_sql_query(question):
#     """
#     Converts a natural language question into an SQL query.
#     Customize this logic based on your database schema and question patterns.
#     """
#     # Example mapping of keywords to columns
#     if "stock" in question.lower():
#         return "SELECT stock_quantity FROM tshirt_priyam"
#     elif "price" in question.lower():
#         return "SELECT price FROM tshirt_priyam"
#     # Add more mappings as needed
#     return None

# def query_database(query):
#     """
#     Executes the given SQL query on the database and returns the results.
#     """
#     try:
#         conn = mysql.connector.connect(
#             user=db_user,
#             password=db_password,
#             host=db_host,
#             database=db_name
#         )
#         cursor = conn.cursor()
#         cursor.execute(query)
#         results = cursor.fetchall()
#         cursor.close()
#         conn.close()

#         # Format results for readability
#         return [str(row) for row in results]

#     except mysql.connector.Error as err:
#         print("MySQL Error:", err)
#         return None
    
    
    
    
    
    
# Cache the SQL query results for repeated questions
@lru_cache(maxsize=100)  # Cache up to 100 queries
def get_cached_sql_query(processed_question):
    return processed_question    
    


  
def generate_sql_query(question, sql_chain, schema):
    
    """
    Generate an SQL query based on the user's question.
    Ensure proper table and column qualification using the database schema.
    """

    try:
        # Preprocess the question
        processed_question = preprocess_input(question)
        
        # Use the SQL chain to generate the query
        ai_response = sql_chain.run({"question": processed_question})

        if not isinstance(ai_response, str):
            raise ValueError("Generated query is not a valid string.")

        # Clean up the query
        sql_query = ai_response.strip()
        sql_query = re.sub(r'(\\n|\\r|\\t)', ' ', sql_query)  # Remove line breaks and tabs
        sql_query = re.sub(r'\s+', ' ', sql_query).strip()
        sql_query = re.sub(r'```', '', sql_query)# Normalize whitespace

        if "content=" in sql_query:
            sql_query = sql_query.split("content=")[-1].strip("'").strip()
            
        if "sql" in sql_query:
            sql_query = sql_query.split("sql")[1].strip("'").strip()  

        if sql_query.lower().startswith("sql"):
            sql_query = sql_query.split("sql", 1)[1].strip("'").strip()

        sql_query = re.sub(r'\+$', '', sql_query)  # Fix for regex pattern
        
       
        # Debugging output
        print(f"Generated Query Before Cleanup: {sql_query}")

        # Ensure all columns are qualified with table names using the schema
        sql_query = qualify_columns_with_table_names(sql_query, schema)
        
        print(f"Generated SQL Query (after qualification): {sql_query}")



        return sql_query
    except Exception as e:
        raise ValueError(f"Error generating SQL query: {str(e)}")



def qualify_columns_with_table_names(query, schema):
    """
    Ensure all columns in the query are qualified with their respective table names.
    Uses the database schema to map columns to tables.
    """
    for table, columns in schema.items():
        for column in columns:
            # Replace unqualified column with table-qualified column if necessary
            pattern = rf'\b{column}\b'  # Match column name as a whole word
            replacement = f"{table}.{column}"
            query = re.sub(pattern, replacement, query)

    print(f"Qualified Query: {query}")  # Debug print to see the qualified query
    return query


def answer_questions(text, conn, sql_chain):
    """Generate responses dynamically using the database schema and queries."""
    questions = extract_questions_from_text(text)
    responses = []

    # Fetch database schema
    schema = get_database_schema(conn)
    print(f"Database Schema: {schema}")  # Debug print for schema

    for question in questions:
        try:
            # Check cache first
            cached_response = get_cached_response(question)
            if cached_response:
                responses.append({
                    "question": question,
                    "response": cached_response
                })
                continue

            # Generate SQL query
            generated_query = generate_sql_query(question, sql_chain, schema)
            print(f"Question: {question}")  # Debug print for the question
            print(f"Generated Query: {generated_query}")  # Debug print for the query

            # Validate the generated SQL query
            validate_sql_query(schema, generated_query)

            # Execute the query
            cursor = conn.cursor()
            cursor.execute(generated_query)
            result = cursor.fetchall()
            cursor.close()

            if result:
                # Format the results into a readable answer
                formatted_answer = "\n".join([", ".join(map(str, row)) for row in result])
            else:
                formatted_answer = "No relevant data found in the database."

            # Cache the result
            cache_response(question, formatted_answer)

            responses.append({
                "question": question,
                "response": formatted_answer
            })
        except Exception as e:
            responses.append({
                "question": question,
                "response": f"Error processing the question: {str(e)}"
            })

    return responses





def extract_questions_from_text(text):
    """Extract potential questions from the text."""
    sentences = text.split(".")
    questions = [sentence.strip() + "?" for sentence in sentences if sentence.strip() and len(sentence.split()) > 5]
    return questions[:5]  # Limit to 5 questions for brevity



@app.route('/analyze_and_summarize', methods=['POST'])
def analyze_and_summarize():
    # Check if the file is part of the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Extract store_id from the form data
    store_id = request.form.get('store_id')  # Use form instead of JSON
    if not store_id or store_id not in DATABASES:
        return jsonify({'error': 'Invalid store ID provided'}), 400

    try:
        # Fetch database configuration
        db_info = DATABASES[store_id]
        db_name_key = f"{store_id}_db_name"
        db_name = db_info.get(db_name_key)

        if not db_name:
            return jsonify({"error": "Database name not configured correctly"}), 400

        # Determine file type and extract text
        file_type = file.content_type
        text = ""

        if file_type == 'application/pdf':
            pdf_reader = PyPDF2.PdfReader(file)
            text = "".join([page.extract_text() for page in pdf_reader.pages])
        elif file_type in ['application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'application/msword']:
            doc = docx.Document(file)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        elif file_type == 'text/plain':
            text = file.read().decode('utf-8')
        elif file_type in ['image/png', 'image/jpeg']:
            image = Image.open(file)
            text = pytesseract.image_to_string(image)
        else:
            return jsonify({'error': 'Unsupported file type'}), 400

        # Generate summary
        summary = text[:500] + "..." if len(text) > 500 else text

        # Connect to the database
        conn = None
        try:
            conn = mysql.connector.connect(
                user=db_info["db_user"],
                password=db_info["db_password"],
                host=db_info["db_host"],
                database=db_name
            )

            # Set up SQL chain and schema retrieval
            schema = get_database_schema(conn)
            prompt_template = generate_sql_prompt_template_with_values(conn, schema)
            sql_prompt = PromptTemplate(input_variables=["question"], template=prompt_template)
            sql_chain = LLMChain(prompt=sql_prompt, llm=llm)

            # Generate responses
            responses = answer_questions(text, conn, sql_chain)

        except mysql.connector.Error as err:
            return jsonify({'error': f'Database error: {str(err)}'}), 500
        finally:
            if conn:
                conn.close()
        
        
        return jsonify({
            'summary': summary,
            'responses': responses
        })
             
        
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500




# @app.route('/refresh-schema', methods=['POST'])
# def refresh_schema():
#   global sql_prompt, sql_chain

#   try:
#         # Retrieve input data from the request
#         if request.is_json:
#             data = request.get_json()
#             new_db_name = data.get('db_name', "").strip()
#             question = data.get('question', "").strip()
#         else:
#             new_db_name = request.form.get('db_name', "").strip()
#             question = request.form.get('question', "").strip()

#         # Validate inputs
#         if not new_db_name:
#             return jsonify({"error": "Database name is required."}), 400
#         if not question:
#             return jsonify({"error": "Question is required."}), 400

#         # Debugging: Log inputs
#         print(f"Database name: {new_db_name}")
#         print(f"Question: {question}")

#         # Connect to the new database
#         conn = mysql.connector.connect(
#          DATABASES
#         )

#         try:
#             # Fetch schema and regenerate the prompt template
#             schema = get_database_schema(conn)
#             dynamic_sql_prompt_template = generate_sql_prompt_template_with_values(schema)
#             sql_prompt = PromptTemplate(
#                 input_variables=["question"],
#                 template=dynamic_sql_prompt_template
#             )
#             sql_chain = sql_prompt | llm  # Recreate the chain with the new prompt

#             # Run the question through the updated chain
#             response = run_prediction(question)  # Replace with your prediction logic
#         finally:
#             # Ensure connection is closed
#             if conn.is_connected():
#                 conn.close()

#         return jsonify({
#             "message": "Schema refreshed successfully!",
#             "question": question,
#             "response": response
#         })

#   except mysql.connector.Error as db_error:
#         print(f"Database error: {db_error}")
#         return jsonify({"error": f"Database error: {db_error}"}), 500
#   except Exception as e:
#         print(f"Unexpected error: {e}")
#         return jsonify({"error": f"Unexpected error: {e}"}), 500




#check stock

@app.route('/checkStock', methods=['POST'])
def check_stock() :
    try:
        # Parse the incoming JSON request
        data = request.json
        store_id = data.get('store_id')
        selected_values = data.get('selected_values', {})

        print("Received Store ID:", store_id)
        print("Received Selected Filters:", selected_values)

        # Validate the store_id
        if store_id not in DATABASES:
            return jsonify({
                "success": False,
                "message": f"Invalid store ID: {store_id}"
            }), 400

        db_info = DATABASES[store_id]
        db_name = db_info.get(f"{store_id}_db_name")

        if not db_name:
            return jsonify({
                "success": False,
                "message": "Database name not configured for the store."
            }), 400

        # Establish a database connection
        conn = mysql.connector.connect(
            host=db_info["db_host"],
            user=db_info["db_user"],
            password=db_info["db_password"],
            database=db_name
        )
        cursor = conn.cursor()

        # Build the query dynamically based on selected filters
        table_name = f"store_{store_id[-1]}"
        where_clauses = []
        query_values = []

        for column_name, column_value in selected_values.items():
            if column_value.strip():  # Ensure the value is not empty
                where_clauses.append(f"{column_name} = %s")
                query_values.append(column_value.strip())

        if not where_clauses:
            return jsonify({
                "success": False,
                "message": "No filters selected. Please choose at least one filter."
            }), 400

        where_clause = " AND ".join(where_clauses)
        sql_query = f"SELECT ID, BrandName, Stock FROM {table_name} WHERE {where_clause}"

        print("Constructed SQL Query:", sql_query)
        print("Query Parameters:", query_values)

        # Execute the query
        cursor.execute(sql_query, tuple(query_values))
        results = cursor.fetchall()

        # Close the database connection
        cursor.close()
        conn.close()

        print("Query Results:", results)

        # If results are found, format them for the response
        if results:
            stock_data = [
                {"product_id": result[0], "brand_name": result[1], "stock_quantity": result[2]}
                for result in results
            ]
            return jsonify({
                "success": True,
                "stock_data": stock_data
            }), 200

        # If no results are found, return a meaningful message
        return jsonify({
            "success": True,
            "stock_data": [],
            "message": "No stock found for the selected criteria."
        }), 200

    except mysql.connector.Error as err:
        print("Database Error:", err)
        return jsonify({
            "success": False,
            "message": f"Database error: {err}"
        }), 500
    except Exception as e:
        print("Unexpected Error:", e)
        return jsonify({
            "success": False,
            "message": f"Unexpected error: {e}"
        }), 500






#stock indicator

# @app.route('/get_stock_quantity', methods=['GET'])
# def get_stock_quantity():
#     # Replace with logic to query your database
#     stock_quantity = 8  # Simulated value
#     return jsonify({'stock_quantity': stock_quantity})

# # Route to refill stock
# @app.route('/refill_stock', methods=['POST'])
# def refill_stock():
#     # Replace with logic to update your database
#     # Assuming stock is refilled successfully
#     return jsonify({'status': 'success', 'message': 'Stock refilled successfully'})



# @app.route('/predict', methods=['POST'])
# def predict():
#     print("Received request")
#     data = request.json

#     # Extract question and store_id from the request
#     question = data.get('question', '')
#     store_id = data.get('store_id', '')

#     if not question:
#         return jsonify({"error": "No question provided"}), 400

#     if not store_id:
#         return jsonify({"error": "No store ID provided"}), 400

#     if store_id not in DATABASES:
#         return jsonify({"error": "Invalid store ID"}), 400

#     # Preprocess the question to normalize it
#     question = preprocess_input(question)

#     # Call the prediction function
#     response = run_prediction(question, store_id)
#     return jsonify({"response": response})
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    print("Received data:", data)  # Debugging log

    question = data.get('question', '').strip()
    store_id = data.get('store_id', '').strip()

    if not question:
        print("Error: No question provided")  # Debugging log
        return jsonify({"error": "No question provided"}), 400

    if not store_id:
        print("Error: store_id is None or empty")  # Debugging log
        return jsonify({"error": "Invalid store ID"}), 400

    # Your prediction logic here...
    try:
        response = run_prediction(question, store_id)
        return jsonify({"response": response})
    except Exception as e:
        print(f"Error during prediction: {e}")  # Debugging log
        return jsonify({"error": "Prediction failed"}), 500


@app.route('/predict1', methods=['POST'])
def predict1():
    print("Received request")
    data = request.json
    question = data.get('question', '')

    if not question:
        return jsonify({"error": "No question provided"}), 400

    # Preprocess the question to normalize it
    question = preprocess_input(question)

    # Call the prediction function
    response = run_prediction1(question)
    return jsonify({"response": response})
    
if __name__ == '__main__':
    app.run(debug=True, port=5000)
