#import time
from flask import Flask, request, render_template, jsonify
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

# Initialize Flask app
app = Flask(__name__)

# Google Gemini AI setup
api_key = "AIzaSyCABdTIiT3IxPur9TU512buK-Cc7Z_GquU"
genai.configure(api_key=api_key)
model = "models/gemini-pro"

try:
    llm = ChatGoogleGenerativeAI(api_key=api_key, model=model)
except Exception as e:
    print("Error initializing Google Generative AI:", e)
    raise

@app.route('/')
def index():
    return render_template('index.html')

# Database connection setup
db_user = "root"
db_password = "userrt"
db_host = "localhost"
db_name = "atliq_tshirts"

try:
    # Initialize the database
    db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}", sample_rows_in_table_info=3)
    db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)
except Exception as e:
    print("Error initializing SQLDatabase:", e)
    raise

# Vectorization setup
try:
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key="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",
        model_name="sentence-transformers/all-MiniLM-l6-v2"
    )
except Exception as e:
    print("Error initializing embeddings:", e)
    raise

cors = CORS()
cors.init_app(app)

# SQL prompt template with better normalization
sql_prompt_template = """
You are an expert SQL query generator. Below is the schema for the t_shirts table:

t_shirts(price, stock_quantity, brand, color, size)

Generate consistent SQL queries for given questions. Ensure no extraneous comments or formatting issues in the output.

Question: {question}

SQL Query:
"""
sql_prompt = PromptTemplate(
    input_variables=["question"],
    template=sql_prompt_template
)

# Create the RunnableSequence
sql_chain = sql_prompt | llm

# Simple cache for query responses
query_cache = {}

def get_cached_response(question):
    hashed_question = sha256(question.encode()).hexdigest()
    return query_cache.get(hashed_question)

def cache_response(question, response):
    hashed_question = sha256(question.encode()).hexdigest()
    query_cache[hashed_question] = response

def preprocess_input(question):
    # Normalize whitespace, punctuation, and make input lowercase
    question = question.strip().lower()  # Convert to lowercase for consistency
    question = re.sub(r'\s+', ' ', question)  # Replace multiple spaces with a single space
    question = re.sub(r'\s*\?$', '', question)  # Remove trailing spaces before question marks
    return question


def run_prediction(question):
    # Check for a cached response
    cached_response = get_cached_response(question)
    if cached_response:
        print("Using cached response")
        return cached_response

    # Initialize sql_query to ensure it's defined in case of an exception
    sql_query = None

    try:
        # Generate SQL query using the custom prompt
        ai_response = sql_chain.invoke({"question": question})

        # Debugging: Inspect the raw AI response
        print("Raw AI Response:", repr(ai_response))  # Debugging: Print raw AI response

        # Extract text if 'text' attribute is available; otherwise, convert directly to string
        sql_query = ai_response.text if hasattr(ai_response, 'text') else str(ai_response)

        # Step 1: Remove metadata if any (e.g., additional_kwargs, response_metadata)
        sql_query = re.sub(r"(additional_kwargs=.*|response_metadata=.*|id=.*|usage_metadata=.*)", '', sql_query)

        # Step 2: Remove any unwanted prefixes like 'sql' at the start of the query
        sql_query = sql_query.lstrip('sql').lstrip()  # Remove 'sql' at the beginning, followed by any leading whitespace


        # Step 3: Remove 'content="```' or any similar unwanted prefixes at the start
        sql_query = re.sub(r'^content="```', '', sql_query)

        # Step 4: Remove unwanted suffixes like ```" or trailing quote characters
        sql_query = re.sub(r'(```"|\"$)', '', sql_query)

        # Step 5: Replace any escaped line breaks (\n, \r, \t) with a single space
        sql_query = re.sub(r'(\\n|\\r|\\t)', ' ', sql_query)

        # Step 6: Condense multiple spaces to a single space and strip any leading/trailing whitespace
        sql_query = re.sub(r'\s+', ' ', sql_query).strip()

        # Debugging: After cleaning, print the cleaned SQL query
        print(" SQL Query:", repr(sql_query))  # Debugging: Print cleaned query

        # Ensure the cleaned sql_query is valid (not empty or malformed)
        if sql_query.lower().startswith("sql"):
         sql_query = sql_query[3:].lstrip()  # Remove 'sql' and any leading whitespace


        # Execute the SQL query using MySQL connector
        conn = mysql.connector.connect(
            user=db_user,
            password=db_password,
            host=db_host,
            database=db_name
        )
        cursor = conn.cursor()
        cursor.execute(sql_query)
        result = cursor.fetchall()

        # Close connection
        cursor.close()
        conn.close()

        # Convert result to JSON-compatible format
        response = {
            "sql_query": sql_query,  # Include the SQL query in the response
            "result": result
        }

        # Cache the response
        cache_response(question, response)
        return response

    except mysql.connector.Error as err:
        print("MySQL Error:", err)
        return {"error": str(err), "sql_query": sql_query if sql_query else "N/A"}
    except TypeError as te:
        print("Type Error:", te)
        return {"error": "Type error: " + str(te), "sql_query": sql_query if sql_query else "N/A"}
    except ValueError as ve:
        print("Value Error:", ve)
        return {"error": "Value error: " + str(ve), "sql_query": sql_query if sql_query else "N/A"}
    except Exception as e:
        print("Error:", e)
        return {"error": "Unexpected error: " + str(e), "sql_query": sql_query if sql_query else "N/A"}


#getstock function

@app.route('/getStock', methods=['POST'])
def get_stock():
    print("Received request for stock check")
    data = request.json

    # Validate input data
    brand = data.get('brand')
    color = data.get('color')
    size = data.get('size')

    if not brand or not color or not size:
        return jsonify({"success": False, "message": "Missing brand, color, or size parameters"}), 400

    # Construct the SQL query to get the stock quantity
    sql_query = f"""
    SELECT stock_quantity 
    FROM t_shirts 
    WHERE brand = '{brand}' AND color = '{color}' AND size = '{size}';
    """

    try:
        # Connect to the database
        conn = mysql.connector.connect(
            user=db_user,
            password=db_password,
            host=db_host,
            database=db_name
        )
        cursor = conn.cursor()
        cursor.execute(sql_query)
        result = cursor.fetchone()

        # Close connection
        cursor.close()
        conn.close()

        if result:
            return jsonify({"success": True, "stock_quantity": result[0]})
        else:
            return jsonify({"success": False, "message": "No stock found for the specified parameters"})

    except mysql.connector.Error as err:
        print("MySQL Error:", err)
        return jsonify({"success": False, "message": str(err)}), 500
    except Exception as e:
        print("Error:", e)
        return jsonify({"success": False, "message": "Unexpected error: " + str(e)}), 500






@app.route('/predict', methods=['POST'])
def predict():
    print("Received request")
    data = request.json
    question = data.get('question', '')

    if not question:
        return jsonify({"error": "No question provided"}), 400

    # Preprocess the question to normalize it
    question = preprocess_input(question)

    # Call the prediction function
    response = run_prediction(question)
    return jsonify({"response": response})


    
if __name__ == '__main__':
    app.run(debug=True, port=5000)
