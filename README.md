**Project Title: LLM-Powered SQL Query Generator**
**Overview:**
Imagine having a personal assistant who can understand your questions about data and give you answers in plain English. That's what this project does! It uses a special type of computer program called a Large Language Model (LLM) to help you ask questions about your data and get the answers you need.

**How it works:**
Ask a question: You type a question in regular English, like "How many t-shirts of size 'XS' from the brand 'Adidas' and color 'black' are available?"
LLM translates: The LLM understands your question and turns it into a special computer language called SQL.

Computer talks to database: The computer uses the SQL to ask your database for the information.
Get the answer: The computer takes the information from the database and translates it back into English so you can understand it.

Connect to your database: Follow the instructions in the download to connect the program to your database.
Start asking questions: Once connected, you can start typing your questions and get answers.
Example:
Question: "How many t-shirts do we have left for Nike in size XS and color white ?"
Answer: "AI ASSISTANT:19."

**Have fun exploring your data!**

**How To Run The Website**
1. Clone the repository:
   git clone https://github.com/abhirup0106/NexusDB.git

 2.Install dependencies:
   pip install -r requirements.txt

3. Start the application:
   python app.py

**Technologies:**
LLM: Specify the GooglePalm, Gemini, SQLDatabaseChain, LLMChain, PromptTemplate and GoogleGeneartiveAi 
Sentence-Transformer: HuggingFace Embeddings
Database driver: The driver used to connect to your database (e.g. mysql-connector-python for MySQL)
Vector Database: ChromaDB

**Acknowledgements:**
Acknowledge any third-party libraries, tools, or resources used in the project.
