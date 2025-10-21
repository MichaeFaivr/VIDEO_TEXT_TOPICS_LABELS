# AI agents for chatbot to respond to user questions by extracting information from various sources.
from http import client
from langchain.agents import create_openai_functions_agent, create_openai_tools_agent, AgentExecutor
from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain.prompts import ChatPromptTemplate
from langchain_experimental.agents import create_csv_agent
"""
from langchain_experimental.vectorstores import FAISS
"""
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel
from langchain.tools import Tool

import pandas as pd
import openai
from openai import OpenAI
import numpy as np
import faiss  # For efficient similarity search

# Additional imports for CSV handling and email sending
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os

from model.constants import COLS_ROWS_CSV_PATH_FILE

load_dotenv() # Load environment variables from .env file
openai.api_key = os.getenv("OPENAI_API_KEY")  # Set your API key as an environment variable


def create_csv_ai_agent(csv_file_path: str) -> any:
    """
    Create an AI agent that can answer questions based on the content of a CSV file.
    Args:
        csv_file_path (str): The path to the CSV file.
    Returns:
        any: An AI agent capable of answering questions about the CSV file.
    """
    print(f"Creating CSV AI agent for file: {csv_file_path}")
    llm = ChatOpenAI(temperature=0)
    
    agent = create_csv_agent(llm, csv_file_path, pandas_kwargs={"dtype": str, "on_bad_lines": "skip", "sep": ",", "quotechar": '"'}, verbose=True, allow_dangerous_code=False)
    return agent


# Code for Research AI Agent from Youtube Tech With Tim Tutorial
class ResearchResponse(BaseModel):
    specificity: str
    ai_model: str
    additional_context: str
    question: str
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

class SpecificAIAgentResponse(ResearchResponse):
    """
    Response model for specific AI agents.
    Input: question about a specific topic.
    Output: structured response with topic, summary, sources, and tools used.
    """
    def __init__(self, **data):
        super().__init__(**data)
        self.specificity = data.get("specificity", "general")
        self.ai_model = data.get("ai_model", "chatgpt-4.1")
        self.additional_context = data.get("additional_context", "")
        self.question = data.get("question", "")
        self.topic = data.get("topic", "")
        self.summary = data.get("summary", "")
        self.sources = data.get("sources", [])
        self.tools_used = data.get("tools_used", [])

    def call_proper_agent(self):
        if self.ai_model == "chatgpt-4.1":
            return create_research_ai_agent_openai()
        elif self.ai_model == "anthropic-claude-3-5-sonnet-20241022":
            return create_research_ai_agent_anthropic()
        elif self.ai_model == "csv-agent":
            return create_csv_ai_agent(self.additional_context)
        elif self.ai_model == "chatgpt-4o-mini":
            if not self.question:
                raise ValueError("Question must be provided for chatgpt-4o-mini model.")
            self.initialize_csv_row_agent()
            return self.ask_csv_research_bot()
        else:
            raise ValueError(f"Unsupported AI model: {self.ai_model}")

    def initialize_csv_row_agent(self):
        """
        Initialize the CSV row agent for the chatgpt-4o-mini model.
        Loads the CSV file, creates embeddings, and builds a FAISS index for similarity search.
        """
        if self.ai_model == "chatgpt-4o-mini":
            # Load your CSV file
            self.df = pd.read_csv(self.additional_context)

            # Combine columns or use whichever you need as your document source
            texts = self.df.apply(lambda x: ' '.join(x.astype(str)), axis=1).tolist()
            print(f"Loaded {len(texts)} rows from CSV")

            self.client = OpenAI()
            # Get embeddings
            embeddings = []
            for text in texts:
                response = self.client.embeddings.create(
                    input=text,
                    model="text-embedding-3-small"
                )
                embeddings.append(response.data[0].embedding)

            embeddings = np.array(embeddings).astype("float32")

            # Build FAISS index for similarity search
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
            self.index.add(embeddings) # index now contains all embeddings and is used in ask_csv_research_bot()
   
    # Chatpt code for answering questions based on a CSV file with rows and columns information
    def ask_csv_research_bot(self, top_k=3) -> str:
        """
        The bot answers questions based on a CSV file using embeddings and similarity search.
        Args:
            query (str): The user's question.
            top_k (int): Number of top relevant rows to consider.
        Returns:
            str: The answer generated by the AI model.
        """
        if self.ai_model != "chatgpt-4o-mini":
            raise ValueError("ask_csv_research_bot can only be used with chatgpt-4o-mini model.")
        
        # Load your CSV file
        df = pd.read_csv(self.additional_context)

        # Combine columns or use whichever you need as your document source
        texts = df.apply(lambda x: ' '.join(x.astype(str)), axis=1).tolist()
        print(f"Loaded {len(texts)} rows from CSV")

        client = OpenAI()
        # Get embeddings
        embeddings = []
        for text in texts:
            response = client.embeddings.create(
                input=text,
                model="text-embedding-3-small"
            )
            embeddings.append(response.data[0].embedding)

        embeddings = np.array(embeddings).astype("float32")

        # Build FAISS index for similarity search
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings) # index now contains all embeddings and is used in ask_csv_research_bot()

        # Create embedding for the query
        query_embedding = client.embeddings.create(
            input=self.question,
            model="text-embedding-3-small"
        ).data[0].embedding

        # Search for most relevant rows
        D, I = index.search(np.array([query_embedding]).astype("float32"), top_k)

        # Collect top matching rows
        context = "\n".join(df.iloc[i].to_string() for i in I[0])

        # Send to GPT model for final answer
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant that answers questions based on provided CSV context."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {self.question}"}
            ]
        )
        return completion.choices[0].message.content


def create_research_ai_agent_anthropic() -> any:
    """
    Create an AI agent that can perform research tasks using the Anthropic API.
    Returns:
        any: An AI agent capable of performing research tasks.
    Required Anthropic API access.
    """
    llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)
    parser = PydanticOutputParser(pydantic_object=ResearchResponse)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """
            You are a research assistant. When asked a question, provide a summary of the topic and relevant sources.
            Question: {query}
            Provide your response in the following format:
            {format_instructions}
            """),
            ("placeholder", "{chat_history}"),
            ("human", "{query}"),
            ("placeholder", "{agent_scratchpad}")
        ]
    ).partial(format_instructions=parser.get_format_instructions())

    ##agent = create_openai_functions_agent(llm, [], prompt)
    agent = create_openai_tools_agent(
        llm=llm,
        tools=[],
        prompt=prompt,
        )
    agent_executor = AgentExecutor(agent=agent, tools=[], verbose=True)
    raw_response = agent_executor.invoke({"query": "Explain the theory of relativity.", "input": "Explain the theory of relativity."})
    try:
        structured_response = parser.parse(raw_response["output"])
        print(structured_response)
        return structured_response
    except Exception as e:
        print(f"Error parsing response: {e}")
        return {"error": str(e)}  


def create_research_ai_agent_openai() -> any:
    """
    Create an AI agent that can perform research tasks using the OpenAI API.
    Returns:
        any: An AI agent capable of performing research tasks.
    Run Error sur quota openai:
    openai.RateLimitError: Error code: 429 - {'error': {'message': 'You exceeded your current quota, 
    please check your plan and billing details. For more information on this error, read the docs: https://platform.openai.com/docs/guides/error-codes/api-errors.', 'type': 'insufficient_quota', 'param': None, 'code': 'insufficient_quota'}}
    """

    llm = ChatOpenAI(model="gpt-4.1", temperature=0)
    parser = PydanticOutputParser(pydantic_object=ResearchResponse)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """
            You are a research assistant. When asked a question, provide a summary of the topic and relevant sources.
            Question: {input}
            Provide your response in the following format:
            {format_instructions}
            """),
            ("placeholder", "{chat_history}"),
            ("human", "{query}"),
            ("placeholder", "{agent_scratchpad}")
        ]
    ).partial(format_instructions=parser.get_format_instructions())

    agent = create_openai_tools_agent(
        llm=llm,
        tools=[],
        prompt=prompt,
    )
    agent_executor = AgentExecutor(agent=agent, tools=[], verbose=True)

    raw_response = agent_executor.invoke({"query": "Explain the theory of relativity.", "input": "Explain the theory of relativity."})

    ##raw_response = agent_executor.run("Explain the theory of relativity.")
    try:
        structured_response = parser.parse(raw_response)["text"]
        print(structured_response)
        return structured_response
    except Exception as e:
        print(f"Error parsing response: {e}")
        return {"error": str(e)}
    


# Chatpt code for answering questions based on a CSV file with rows and columns information
def ask_csv_research_bot_draft(query, top_k=3) -> str:
    """
    The bot answers questions based on a CSV file using embeddings and similarity search.
    Args:
        query (str): The user's question.
        top_k (int): Number of top relevant rows to consider.
    Returns:
        str: The answer generated by the AI model.
    """
    # Create embedding for the query
    query_embedding = client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    ).data[0].embedding

    # Search for most relevant rows
    D, I = index.search(np.array([query_embedding]).astype("float32"), top_k)

    # Collect top matching rows
    context = "\n".join(df.iloc[i].to_string() for i in I[0])

    # Send to GPT model for final answer
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant that answers questions based on provided CSV context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
    )
    return completion.choices[0].message.content

    

def send_email_function(to_email: str, subject: str, body: str) -> str:
    """
    Send an email using SMTP.
    Args:
        to_email (str): Recipient email address
        subject (str): Email subject
        body (str): Email body content     
    Returns:
        str: Success or error message
    """
    try:
        smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
        sender_email = os.getenv("SENDER_EMAIL")
        sender_password = os.getenv("SENDER_PASSWORD")
        
        if not sender_email or not sender_password:
            return "Error: SENDER_EMAIL and SENDER_PASSWORD environment variables must be set"
        
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        
        return f"Email sent successfully to {to_email}"
    except Exception as e:
        return f"Error sending email: {str(e)}"


def create_email_ai_agent() -> any:
    """
    Create an AI agent that can send emails.
    Returns:
        any: An AI agent capable of sending emails.
    """
    llm = ChatOpenAI(temperature=0)
    
    email_tool = Tool(
        name="send_email",
        description="Send an email. Input should be a string with format 'to_email|subject|body'",
        func=lambda input_str: send_email_function(*input_str.split('|', 2))
    )
    
    prompt = ChatPromptTemplate.from_template(
        "You are an email assistant. Use the send_email tool to send emails.\n\n"
        "When using the tool, format the input as: recipient_email|subject|body\n\n"
        "Question: {input}\n\n"
        "Use the following format:\n"
        "Thought: I need to send an email\n"
        "Action: send_email\n"
        "Action Input: email@example.com|Subject Here|Email body content here\n"
        "Observation: [result of sending email]\n"
    )
    
    agent = create_openai_functions_agent(llm, [email_tool], prompt)
    return agent
