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

# Google Gemini AI setup
api_key = "AIzaSyDWW5LQbReUDSqJbQnS0XXouWD_iI6B-lI"
genai.configure(api_key=api_key)
model = "models/gemini-pro"

try:
    llm = ChatGoogleGenerativeAI(api_key=api_key, model=model)
except Exception as e:
    print("Error initializing Google Generative AI:", e)
    raise




@app.route('/')
def home():
    return render_template('revolve.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        entered_captcha = request.form.get('captchaInput')
        actual_captcha = session.get('captcha')

        # Check CAPTCHA first
        if entered_captcha != actual_captcha:
            error = 'Incorrect CAPTCHA. Please try again.'
            return render_template('login.html', captcha=session['captcha'], error=error)

        # Verify credentials
        if email in USER_CREDENTIALS and USER_CREDENTIALS[email] == password:
            session['logged_in'] = True  # Set session flag for successful login
            return redirect(url_for('index'))
        else:
            error = 'Invalid email or password.'
            return render_template('login.html', captcha=session['captcha'], error=error)

    # Generate a new CAPTCHA for the login page
    session['captcha'] = generate_captcha()
    return render_template('login.html', captcha=session['captcha'])

@app.route('/index')
def index():
    if not session.get('logged_in'):  # Check if the user is logged in
        return redirect(url_for('login'))  # Redirect to login if not authenticated
    return render_template('index.html')  # Render the protected page

@app.route('/logout')
def logout():
    session.pop('logged_in', None)  # Remove the logged-in flag from the session
    return redirect(url_for('login'))  # Redirect to login after logout


@app.route('/check')
def check():
    session.pop('logged_in', None)  # Remove the logged-in flag from the session
    return redirect(url_for('login'))



def generate_captcha():
    import random
    import string
    return ''.join(random.choices(string.ascii_letters + string.digits, k=6))

@app.route('/refresh-captcha')
def refresh_captcha():
    captcha = generate_captcha()  # Generate a new CAPTCHA
    session['captcha'] = captcha  # Store it in the session
    return {'captcha': captcha}  # Return it as JSON

@app.route('/index1')
def index1():
    return render_template('index1.html')







# Database connection setup
db_user = "root"
db_password = "userrt"
db_host = "localhost"
db_name = "flut"

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
    schema = {}
    cursor = conn.cursor()
    cursor.execute("SHOW TABLES")
    tables = cursor.fetchall()
    for table in tables:
        table_name = table[0]
        cursor.execute(f"DESCRIBE {table_name}")
        columns = cursor.fetchall()
        schema[table_name] = [column[0] for column in columns]
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
2. Use the `Category` column for broader classifications (e.g., WomensWear).
3. Ensure consistency across similar queries.

Example questions and SQL queries:
- Question: "What is the maximum stock for sarees?"
  SQL Query: SELECT MAX(Stock) AS MaximumStock FROM fashionhub_products WHERE Type = 'Sarees';
- Question: "What is the stock for shirts?"
  SQL Query: SELECT Stock FROM fashionhub_products WHERE Type = 'Shirts';

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
def run_prediction(question):
    question = preprocess_input(question)

    # Check for cached response
    cached_response = get_cached_response(question)
    if cached_response:
        print("Using cached response")
        return cached_response

    # Initialize variables
    sql_query = None

    try:
        # Generate SQL query using the custom prompt
        ai_response = sql_chain.invoke({"question": question})

        # Extract text if 'text' attribute is available; otherwise, convert directly to string
        sql_query = ai_response.text if hasattr(ai_response, 'text') else str(ai_response)

        #Clean the generated SQL query
        sql_query = re.sub(r"(additional_kwargs=.*|response_metadata=.*|id=.*|usage_metadata=.*)", '', sql_query)
        sql_query = re.sub(r'^content="```', '', sql_query)
        sql_query = re.sub(r'(```"|\"$)', '', sql_query)
        sql_query = re.sub(r'(\\n|\\r|\\t)', ' ', sql_query)
        sql_query = re.sub(r'\s+', ' ', sql_query).strip()
        sql_query = sql_query.lstrip('sql').lstrip()


         
        if "content=" in sql_query:
                sql_query = sql_query.split("content=")[1].strip("'").strip()
                
        if "content=" in sql_query:
                sql_query = sql_query.split("content=")[-1].strip("'").strip()  
                     
        if "sql" in sql_query:
                sql_query = sql_query.split("sql")[1].strip("'").strip()  
                
        # if "" in sql_query:
        #        sql_query = re.sub(r'(")', '', sql_query).strip("'").strip()  
        
        sql_query = re.sub(r'`+$', '', sql_query) 





        # Ensure valid SQL query
        validate_sql_query(schema, sql_query)

        # Execute the SQL query
        cursor = conn.cursor()
        cursor.execute(sql_query)
        result = cursor.fetchall()

        # Close connection
        cursor.close()

        # Prepare response
        response = {
            "sql_query": sql_query,
            "result": result
        }

        # Cache and return the response
        cache_response(question, response)
        return response

    except mysql.connector.Error as err:
        print("MySQL Error:", err)
        return {"error": str(err), "sql_query": sql_query if sql_query else "N/A"}
    except ValueError as ve:
        print("Value Error:", ve)
        return {"error": "Value error: " + str(ve), "sql_query": sql_query if sql_query else "N/A"}
    except Exception as e:
        print("Unexpected Error:", e)
        return {"error": "Unexpected error: " + str(e), "sql_query": sql_query if sql_query else "N/A"}

# Main execution
try:
    # Connect to the database
    conn = mysql.connector.connect(
        user=db_user,
        password=db_password,
        host=db_host,
        database=db_name
    )

    # Fetch schema and generate prompt template
    schema = get_database_schema(conn)
    dynamic_sql_prompt_template = generate_sql_prompt_template_with_values(conn, schema)

    # Initialize the SQL prompt
    sql_prompt = PromptTemplate(
        input_variables=["question"],
        template=dynamic_sql_prompt_template
    )

    # Create the SQL generation pipeline
    sql_chain = sql_prompt | llm

except Exception as e:
    print("Error initializing SQLDatabase or prompt template:", e)
    raise






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
        column_names = data.get('column_names', [])  # Default to empty list if not provided

        # Connect to the database
        conn = mysql.connector.connect(
            user=db_user,
            password=db_password,
            host=db_host,
            database=db_name
        )
        cursor = conn.cursor()

        # Dynamically fetch all column names from the database table
        cursor.execute("SHOW COLUMNS FROM fashionhub_products;")
        db_columns = [row[0].strip() for row in cursor.fetchall()]  # Strip spaces for consistency

        # Log the fetched columns for debugging
        print("Fetched columns from DB:", db_columns)

        # Ensure 'Stock_Quantity' and any "Stock" columns are completely excluded
        excluded_columns = ['stock_quantity', 'stock']  # Add 'stock' explicitly to exclusion
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
            # Skip any excluded columns
            if column_name.lower() in excluded_columns:
                continue
            sql_query = f"SELECT DISTINCT {column_name} FROM fashionhub_products;"
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
        selected_values = request.json or {}

        # Connect to the database
        conn = mysql.connector.connect(
            user=db_user,
            password=db_password,
            host=db_host,
            database=db_name
        )
        cursor = conn.cursor()

        # Build the query to fetch stock availability based on selected values
        filters = []
        for column, value in selected_values.items():
            if value:  # Only add filters for selected values
                filters.append(f"{column} = '{value}'")

        # Create WHERE clause from filters
        where_clause = " AND ".join(filters) if filters else "1=1"
        query = f"SELECT BrandName, Stock FROM fashionhub_products WHERE {where_clause};"

        cursor.execute(query)
        results = cursor.fetchall()

        # Close the connection
        cursor.close()
        conn.close()

        # Parse the results to find available combinations
        available_combinations = {}
        for row in results:
            brand_name, stock_quantity = row
            if stock_quantity > 0:
                available_combinations[brand_name] = True  # Mark as available

        return jsonify({"success": True, "available_combinations": available_combinations}), 200

    except mysql.connector.Error as err:
        print("MySQL Error:", err)
        return jsonify({"success": False, "message": str(err)}), 500
    except Exception as e:
        print("Error:", e)
        return jsonify({"success": False, "message": "Unexpected error: " + str(e)}), 500

    

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
    
    
    


#modify tshirt

@app.route('/tshirt', methods=['POST'])
def modify_tshirt():
    try:
        # Log the incoming request for debugging
        app.logger.debug(f"Incoming request: {request.json}")

        # Get the action ('add' or 'remove') and the data from the JSON request
        action = request.json.get('action')
        data = request.json.get('data')

        # Log the action and data to ensure they are correct
        app.logger.debug(f"Action: {action}, Data: {data}")

        # Check if the necessary fields are present in the data
        if not data or not data.get('brand') or not data.get('color') or not data.get('size') or not data.get('stock_quantity'):
            app.logger.warning(f"Missing required fields in data: {data}")
            return jsonify({"error": "Missing required fields in the data."}), 400

        # Connect to the MySQL database
        conn = mysql.connector.connect(
            host=db_host,  # Replace with your database host
            user=db_user,  # Replace with your database user
            password=db_password,  # Replace with your database password
            database=db_name  # Replace with your database name
        )
        cursor = conn.cursor()

        # Check if the item already exists for 'add' action
        if action == 'add':
            query = """
                SELECT stock_quantity FROM t_shirts WHERE brand = %s AND color = %s AND size = %s
            """
            cursor.execute(query, (data['brand'], data['color'], data['size']))
            result = cursor.fetchone()

            if result:  # If the entry exists
                # Update stock_quantity only, do not update price
                new_stock_quantity = result[0] + int(data['stock_quantity'])  # Add the new stock to the existing stock
                update_query = """
                    UPDATE t_shirts
                    SET stock_quantity = %s
                    WHERE brand = %s AND color = %s AND size = %s
                """
                cursor.execute(update_query, (new_stock_quantity, data['brand'], data['color'], data['size']))
                app.logger.info(f"Updated stock for existing T-shirt: {data['brand']} - {data['color']} - {data['size']} - New Stock: {new_stock_quantity}")
            else:
                # If the item doesn't exist, insert a new entry with price and stock
                insert_query = """
                    INSERT INTO t_shirts (brand, color, size, stock_quantity, price)
                    VALUES (%s, %s, %s, %s, %s)
                """
                cursor.execute(insert_query, (data['brand'], data['color'], data['size'], data['stock_quantity'], data['price']))
                app.logger.info(f"Added new T-shirt: {data['brand']} - {data['color']} - {data['size']} - Price: {data['price']}")

        # Handle the 'remove' action
        elif action == 'remove':
            # Query to check if the item exists
            query = """
                SELECT stock_quantity FROM t_shirts WHERE brand = %s AND color = %s AND size = %s
            """
            cursor.execute(query, (data['brand'], data['color'], data['size']))
            result = cursor.fetchone()

            if result:  # If the entry exists
                current_stock_quantity = result[0]
                new_stock_quantity = current_stock_quantity - int(data['stock_quantity'])  # Subtract the stock quantity

                if new_stock_quantity > 0:
                    # Update the stock quantity if it's greater than zero
                    update_query = """
                        UPDATE t_shirts
                        SET stock_quantity = %s
                        WHERE brand = %s AND color = %s AND size = %s
                    """
                    cursor.execute(update_query, (new_stock_quantity, data['brand'], data['color'], data['size']))
                    app.logger.info(f"Reduced stock for T-shirt: {data['brand']} - {data['color']} - {data['size']} - New Stock: {new_stock_quantity}")
                else:
                    # Remove the item completely if the stock becomes zero or negative
                    delete_query = """
                        DELETE FROM t_shirts WHERE brand = %s AND color = %s AND size = %s
                    """
                    cursor.execute(delete_query, (data['brand'], data['color'], data['size']))
                    app.logger.info(f"Removed T-shirt: {data['brand']} - {data['color']} - {data['size']} due to zero or negative stock.")
            else:
                # Log if the item does not exist
                app.logger.warning(f"T-shirt not found: {data['brand']} - {data['color']} - {data['size']}")
                return jsonify({"error": "T-shirt not found. Cannot remove stock."}), 404

        # Commit changes to the database
        conn.commit()

        return jsonify({"message": f"Successfully {action}ed data!"}), 200

    except mysql.connector.Error as db_err:
        # Log and handle MySQL database errors
        app.logger.error(f"Database error: {db_err}")
        return jsonify({"error": f"Database error: {str(db_err)}"}), 500

    except Exception as e:
        # Log and handle general errors
        app.logger.error(f"Error occurred: {e}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

    finally:
        # Ensure that the database connection is closed
        if 'cursor' in locals() and cursor:
            cursor.close()
        if 'conn' in locals() and conn:
            conn.close()



#getstock function

# @app.route('/getStock', methods=['POST'])
# def get_stock():
#     print("Received request for stock check")
#     data = request.json

#     # Validate input data
#     brand = data.get('brand')
#     color = data.get('color')
#     size = data.get('size')

#     if not brand or not color or not size:
#         return jsonify({"success": False, "message": "Missing brand, color, or size parameters"}), 400

#     # Construct the SQL query to get the stock quantity
#     sql_query = f"""
#     SELECT stock_quantity 
#     FROM tshirt_priyam
#     WHERE brand = '{brand}' AND color = '{color}' AND size = '{size}';
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
#         cursor.execute(sql_query)
#         result = cursor.fetchone()

#         # Close connection
#         cursor.close()
#         conn.close()

#         if result:
#             return jsonify({"success": True, "stock_quantity": result[0]})
#         else:
#             return jsonify({"success": False, "message": "No stock found for the specified parameters"})

#     except mysql.connector.Error as err:
#         print("MySQL Error:", err)
#         return jsonify({"success": False, "message": str(err)}), 500
#     except Exception as e:
#         print("Error:", e)
#         return jsonify({"success": False, "message": "Unexpected error: " + str(e)}), 500


#pdf

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
    
  
def generate_sql_query(question, sql_chain):
    """Generate an SQL query dynamically using the provided SQL chain."""
    try:
        # Preprocess the question
        processed_question = preprocess_input(question)

        # Use the SQL chain to generate the query
        ai_response = sql_chain.run({"question": processed_question})

        # Ensure the response is a valid string
        if not isinstance(ai_response, str):
            raise ValueError("Generated query is not a valid string.")
        
        
          

        # Clean up the query
        sql_query = ai_response.strip()
        sql_query = re.sub(r'(\\n|\\r|\\t)', ' ', sql_query)  # Remove line breaks and tabs
        sql_query = re.sub(r'\s+', ' ', sql_query).strip()  # Normalize whitespace
        
       
        
        if "content=" in sql_query:
                sql_query = sql_query.split("content=")[1].strip("'").strip()
                
        if "content=" in sql_query:
                sql_query = sql_query.split("content=")[-1].strip("'").strip()  
                     
        if "sql" in sql_query:
                sql_query = sql_query.split("sql")[1].strip("'").strip()  
                
        # if "" in sql_query:
        #        sql_query = re.sub(r'(")', '', sql_query).strip("'").strip()  
        
        sql_query = re.sub(r'`+$', '', sql_query) 
        
     



        return sql_query
    except Exception as e:
        raise ValueError(f"Error generating SQL query: {str(e)}")
    
   

def answer_questions(text, conn, sql_chain):
    """Generate responses dynamically using the database schema and queries."""
    questions = extract_questions_from_text(text)
    responses = []

    # Fetch database schema
    schema = get_database_schema(conn)

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
            generated_query = generate_sql_query(question, sql_chain)

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
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Handle file types and extract text
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

        # Connect to database
        conn = mysql.connector.connect(
            user=db_user,
            password=db_password,
            host=db_host,
            database=db_name
        )

        # Set up SQL chain
        schema = get_database_schema(conn)
        prompt_template = generate_sql_prompt_template_with_values(conn, schema)
        sql_prompt = PromptTemplate(input_variables=["question"], template=prompt_template)
        sql_chain = LLMChain(prompt=sql_prompt, llm=llm)

        # Generate responses
        responses = answer_questions(text, conn, sql_chain)

        # Close connection
        conn.close()

        return jsonify({
            'summary': summary,
            'responses': responses
        })

    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500



@app.route('/refresh-schema', methods=['POST'])
def refresh_schema():
  global sql_prompt, sql_chain

  try:
        # Retrieve input data from the request
        if request.is_json:
            data = request.get_json()
            new_db_name = data.get('db_name', "").strip()
            question = data.get('question', "").strip()
        else:
            new_db_name = request.form.get('db_name', "").strip()
            question = request.form.get('question', "").strip()

        # Validate inputs
        if not new_db_name:
            return jsonify({"error": "Database name is required."}), 400
        if not question:
            return jsonify({"error": "Question is required."}), 400

        # Debugging: Log inputs
        print(f"Database name: {new_db_name}")
        print(f"Question: {question}")

        # Connect to the new database
        conn = mysql.connector.connect(
            user=db_user,
            password=db_password,
            host=db_host,
            database=db_name
        )

        try:
            # Fetch schema and regenerate the prompt template
            schema = get_database_schema(conn)
            dynamic_sql_prompt_template = generate_sql_prompt_template_with_values(schema)
            sql_prompt = PromptTemplate(
                input_variables=["question"],
                template=dynamic_sql_prompt_template
            )
            sql_chain = sql_prompt | llm  # Recreate the chain with the new prompt

            # Run the question through the updated chain
            response = run_prediction(question)  # Replace with your prediction logic
        finally:
            # Ensure connection is closed
            if conn.is_connected():
                conn.close()

        return jsonify({
            "message": "Schema refreshed successfully!",
            "question": question,
            "response": response
        })

  except mysql.connector.Error as db_error:
        print(f"Database error: {db_error}")
        return jsonify({"error": f"Database error: {db_error}"}), 500
  except Exception as e:
        print(f"Unexpected error: {e}")
        return jsonify({"error": f"Unexpected error: {e}"}), 500




#check stock

@app.route('/checkStock', methods=['POST'])
def checkStock():
    try:
        data = request.json
        selected_values = data.get('selected_values', {})

        # Log selected values for debugging
        print("Selected Values:", selected_values)

        # Build the WHERE clause dynamically based on non-empty selected values
        where_clauses = []
        values = []

        for column_name, value in selected_values.items():
            if value and value != '':  # Only include the column if a value is selected
                where_clauses.append(f"{column_name} = %s")
                values.append(value)

        if not where_clauses:
            return jsonify({"success": False, "message": "No criteria selected to check stock"}), 400

        # Create the SQL query dynamically
        sql_query = (
            "SELECT ID, BrandName, Stock "
            "FROM fashionhub_products WHERE " + " AND ".join(where_clauses)
        )

        # Connect to the database and execute the query
        conn = mysql.connector.connect(
            user=db_user,
            password=db_password,
            host=db_host,
            database=db_name
        )
        cursor = conn.cursor()
        cursor.execute(sql_query, tuple(values))
        results = cursor.fetchall()

        cursor.close()
        conn.close()

        if results:
            # Structure the data to include product_id, brand_name, and stock_quantity
            stock_data = [
                {"product_id": result[0], "brand_name": result[1], "stock_quantity": result[2]}
                for result in results
            ]
            return jsonify({"success": True, "stock_data": stock_data}), 200
        else:
            # Return empty stock data with success: true
            return jsonify({"success": True, "stock_data": [], "message": "No stock found for the selected criteria"}), 200

    except mysql.connector.Error as err:
        print("MySQL Error:", err)
        return jsonify({"success": False, "message": str(err)}), 500
    except Exception as e:
        print("Error:", e)
        return jsonify({"success": False, "message": "Unexpected error: " + str(e)}), 500





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
