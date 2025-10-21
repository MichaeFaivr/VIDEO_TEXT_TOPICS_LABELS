# AI agents for chatbot to respond to user questions by extracting information from various sources.
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
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os

load_dotenv() # Load environment variables from .env file


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
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

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
