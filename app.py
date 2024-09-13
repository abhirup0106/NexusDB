from flask import Flask, request, render_template ,jsonify
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAI
from langchain_experimental.sql import SQLDatabaseChain
from langchain.utilities import SQLDatabase
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import mysql.connector



import pymysql
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings

import os
from dotenv import  load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import (
    PromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

# Initialize Flask app
app = Flask(__name__)

# Google Gemini AI setup
api_key = "AIzaSyAtOQvs295YhpK23aR8FiUbC3z_8-_JbPM"
genai.configure(api_key=api_key)
#model = genai.GenerativeModel('gemini-pro')     # Use GoogleGenAI from langchain-google-genai
model = "models/gemini-pro"
llm = ChatGoogleGenerativeAI(api_key=api_key, model=model)


    
@app.route('/')
def index():
    return render_template('index.html')



# Database connection setup
db_user = "root"
db_password = "userrt"
db_host = "localhost"
db_name = "atliq_tshirts"

# Initialize the database
#db_connection_string = f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}"
db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}",sample_rows_in_table_info=3)
db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)

# Vectorization setup
embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key="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",
        model_name="sentence-transformers/all-MiniLM-l6-v2"
    )





sql_prompt_template = """
You are an expert SQL query generator. Below is the schema for the t_shirts table:

t_shirts(price, stock_quantity, brand)

Given a natural language input, generate a correct SQL query without any prefixes or comments, only valid SQL.

Question: {question}

SQL Query:
"""





sql_prompt = PromptTemplate(
    input_variables=["question"],
    template=sql_prompt_template
)

sql_chain = LLMChain(llm=llm, prompt=sql_prompt)



def run_prediction(question):
    try:
        # Generate SQL query using the custom prompt
        sql_query = sql_chain.run(question)

        # Log the raw SQL query to understand the formatting issues
        print("Raw SQL Query Generated:", sql_query)
        
        # Clean up the SQL query by removing unwanted formatting
        sql_query = sql_query.replace('```sql', '').replace('```', '').replace('```', '').strip()
        sql_query = sql_query.replace("\n", " ").strip()

        print("Cleaned SQL Query:", sql_query)
        
        # Directly execute the SQL query using MySQL connector
        conn = mysql.connector.connect(
            user=db_user,
            password=db_password,
            host=db_host,
            database=db_name
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
    


# Flask route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    

    print("Received request")
    data = request.json
    question = data.get('question', '')
    
    if not question:
        return jsonify({"error": "No question provided"}), 400
    
    # Call the prediction function
    response = run_prediction(question)
    return jsonify({"response": response})


if __name__ == '__main__':
app.run(host='0.0.0.0', port=8080)



