# ----------------------------------------------------------------------------------------------------
# AI Job Assistant
# Service 1: Marketstack API
# Service 2: Semantic Query using Job Search Tracker + ChromaDB
# Service 3: LinkedIn MCP Server Connection
# Includes Guardrails + GPT response formatting
# ----------------------------------------------------------------------------------------------------

import os
import re
import asyncio
import requests
import pandas as pd
import chromadb

from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


load_dotenv()

# ---------------- CONFIG ----------------

CSV_PATH = "data/Job_Search_Tracker.csv"
CHROMA_PATH = "chroma_job_tracker"
COLLECTION_NAME = "job_search_tracker"

PROJECT_ROOT = r"C:\Users\h_daw\dsiai\deploying-ai"

NPX_COMMAND = (
    r"C:\Program Files\nodejs\npx.cmd"
    if os.path.exists(r"C:\Program Files\nodejs\npx.cmd")
    else "npx"
)

OPENAI_MODEL = "gpt-4.1-mini"

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ---------------- GUARDRAILS ----------------

RESTRICTED_TOPICS = [
    "cat",
    "cats",
    "dog",
    "dogs",
    "horoscope",
    "horoscopes",
    "zodiac",
    "taylor swift",
]

SYSTEM_PROMPT_PATTERNS = [
    "system prompt",
    "reveal your prompt",
    "show your prompt",
    "ignore previous instructions",
    "change your instructions",
    "modify your system prompt",
    "developer message",
    "hidden instructions",
]


class Guardrails:
    @staticmethod
    def check(user_input: str):
        text = user_input.lower()

        for phrase in SYSTEM_PROMPT_PATTERNS:
            if phrase in text:
                return False, "I cannot reveal, modify, or discuss hidden system instructions."

        for topic in RESTRICTED_TOPICS:
            if re.search(rf"\b{re.escape(topic)}\b", text):
                return False, "I cannot respond to questions about that restricted topic."

        return True, None


# ---------------- GPT FORMATTER ----------------

def format_with_gpt(raw_text: str, context="general"):
    if not os.getenv("OPENAI_API_KEY"):
        return raw_text

    try:
        if context == "market":
            prompt = (
                "You are a financial assistant. Rewrite the following stock data "
                "into a short, natural, easy-to-understand summary for a user. "
                "Keep it concise and professional.\n\n"
                f"{raw_text}"
            )
        else:
            prompt = (
                "Rewrite the following response in a clean, readable, professional format. "
                "Do not add new information.\n\n"
                f"{raw_text}"
            )

        response = openai_client.responses.create(
            model="gpt-4.1-mini",
            input=prompt
        )

        return response.output_text

    except Exception:
        return raw_text


# ---------------- SERVICE 1: MARKETSTACK API ----------------

class MarketstackService:
    def __init__(self):
        self.api_key = os.getenv("MARKETSTACK_API_KEY")
        self.url = "http://api.marketstack.com/v1/eod/latest"

    def get_stock(self, symbol="AAPL"):
        if not self.api_key:
            return (
                "Marketstack API key is missing.\n\n"
                "Add this to your .env file:\n"
                "MARKETSTACK_API_KEY=your_marketstack_api_key"
            )

        params = {
            "access_key": self.api_key,
            "symbols": symbol.upper()
        }

        try:
            response = requests.get(self.url, params=params, timeout=20)
            
            if response.status_code != 200:
                return f"Marketstack API error: {response.status_code} - {response.text}"

            data = response.json().get("data", [])

            if not data:
                return f"No market data found for symbol: {symbol}"

            stock = data[0]

            return f"""
Marketstack API Result

Stock Symbol: {stock.get("symbol", symbol.upper())}
Close Price: {stock.get("close", "Not provided")}
Open Price: {stock.get("open", "Not provided")}
High Price: {stock.get("high", "Not provided")}
Low Price: {stock.get("low", "Not provided")}
Volume: {stock.get("volume", "Not provided")}
Date: {stock.get("date", "Not provided")}
""".strip()

        except Exception as error:
            return f"Marketstack API call failed: {error}"


# ---------------- SERVICE 2: SEMANTIC JOB TRACKER ----------------

class JobSemanticService:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = self.client.get_or_create_collection(name=COLLECTION_NAME)
        self.df = self.load_csv()

    def load_csv(self):
        if not os.path.exists(CSV_PATH):
            raise FileNotFoundError(
                f"CSV file not found at {CSV_PATH}. "
                "Place Job_Search_Tracker.csv inside the data folder."
            )

        df = pd.read_csv(CSV_PATH)

        df = df.drop(
            columns=[col for col in df.columns if "Unnamed" in col],
            errors="ignore"
        )

        df = df.fillna("Not provided")
        return df

    def row_to_document(self, row):
        return f"""
Company: {row.get("Company Name", "Not provided")}
Job Title: {row.get("Job Title", "Not provided")}
Location: {row.get("Location", "Not provided")}
Date Applied: {row.get("Date Applied", "Not provided")}
Interview: {row.get("Receive Interview", "Not provided")}
Application Status: {row.get("Application Status", "Not provided")}
Follow-up Date: {row.get("Follow-up Date", "Not provided")}
How Applied: {row.get("How Applied", "Not provided")}
Salary: {row.get("Salary", "Not provided")}
ATS Score: {row.get("ATS Score", "Not provided")}
Notes: {row.get("Notes", "Not provided")}
Job Link: {row.get("Job Posting Link", "Not provided")}
""".strip()

    def build_vector_database(self):
        if self.collection.count() > 0:
            print("Semantic database already exists. Skipping rebuild.")
            return

        documents = []
        ids = []
        metadatas = []

        for index, row in self.df.iterrows():
            document = self.row_to_document(row)

            documents.append(document)
            ids.append(str(index))
            metadatas.append({
                "company": str(row.get("Company Name", "Not provided")),
                "job_title": str(row.get("Job Title", "Not provided")),
                "source": str(row.get("How Applied", "Not provided")),
                "interview": str(row.get("Receive Interview", "Not provided")),
                "status": str(row.get("Application Status", "Not provided")),
            })

        embeddings = self.model.encode(documents).tolist()

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

        print(f"Added {len(documents)} job records to ChromaDB.")

    def list_applied_jobs(self):
        required = ["Company Name", "Job Title", "How Applied", "Receive Interview"]

        for column in required:
            if column not in self.df.columns:
                return f"The CSV does not contain a required column: {column}"

        jobs = (
            self.df[required]
            .replace("Not provided", pd.NA)
            .dropna(subset=["Job Title"])
        )

        response = f"You applied for {len(jobs)} job application(s):\n\n"

        for i, row in jobs.iterrows():
            response += (
                f"{i + 1}. {row.get('Job Title', 'Not provided')}\n"
                f"   Company: {row.get('Company Name', 'Not provided')}\n"
                f"   Source: {row.get('How Applied', 'Not provided')}\n"
                f"   Interview: {row.get('Receive Interview', 'Not provided')}\n\n"
            )

        return response

    def interview_summary(self):
        if "Receive Interview" not in self.df.columns:
            return "The CSV does not contain a 'Receive Interview' column."

        interview_df = self.df[
            self.df["Receive Interview"]
            .astype(str)
            .str.lower()
            .str.contains("yes", na=False)
        ]

        if interview_df.empty:
            return "You have not recorded any interviews yet."

        response = f"You received {len(interview_df)} interview(s):\n\n"

        for i, row in interview_df.iterrows():
            response += (
                f"{i + 1}. {row.get('Job Title', 'Not provided')}\n"
                f"   Company: {row.get('Company Name', 'Not provided')}\n"
                f"   Source: {row.get('How Applied', 'Not provided')}\n"
                f"   Status: {row.get('Application Status', 'Not provided')}\n\n"
            )

        return response

    def source_success_summary(self):
        if "How Applied" not in self.df.columns:
            return "The CSV does not contain a 'How Applied' column."

        if "Receive Interview" not in self.df.columns:
            return "The CSV does not contain a 'Receive Interview' column."

        df = self.df.copy()
        df["source_clean"] = df["How Applied"].astype(str).str.lower()

        def normalize_source(value):
            if "linkedin" in value:
                return "LinkedIn"
            if "indeed" in value:
                return "Indeed"
            if "glassdoor" in value:
                return "Glassdoor"
            if "company" in value or "website" in value:
                return "Company Website"
            return "Other / Not Provided"

        df["source_group"] = df["source_clean"].apply(normalize_source)

        df["got_interview"] = (
            df["Receive Interview"]
            .astype(str)
            .str.lower()
            .str.contains("yes", na=False)
        )

        summary = (
            df.groupby("source_group")
            .agg(
                applications=("Job Title", "count"),
                interviews=("got_interview", "sum")
            )
            .reset_index()
        )

        summary["interview_rate"] = (
            summary["interviews"] / summary["applications"] * 100
        ).round(1)

        summary = summary.sort_values(
            by=["interviews", "interview_rate"],
            ascending=False
        )

        response = "Application success by source:\n\n"

        for _, row in summary.iterrows():
            response += (
                f"- {row['source_group']}: "
                f"{row['applications']} application(s), "
                f"{row['interviews']} interview(s), "
                f"{row['interview_rate']}% interview rate\n"
            )

        best = summary.iloc[0]
        response += f"\nMost successful source so far: {best['source_group']}"

        return response

    def semantic_search(self, question, top_k=5):
        query_embedding = self.model.encode([question]).tolist()[0]

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        response = "Most relevant job records:\n\n"

        for i, document in enumerate(results["documents"][0], start=1):
            response += f"Result {i}\n{document}\n\n{'-' * 50}\n\n"

        return response

    def answer(self, question):
        q = question.lower()

        if (
            "which jobs" in q
            or "jobs did i apply" in q
            or "job titles" in q
            or "job title" in q
            or "applied for" in q
        ):
            return self.list_applied_jobs()

        if "interview" in q:
            return self.interview_summary()

        if (
            "indeed" in q
            or "glassdoor" in q
            or "source" in q
            or "site" in q
            or "successful" in q
        ):
            return self.source_success_summary()

        return self.semantic_search(question)


# ---------------- SERVICE 3: LINKEDIN MCP ----------------

class LinkedInMCPService:
    def __init__(self):
        self.server = StdioServerParameters(
            command=NPX_COMMAND,
            args=[
                "-y",
                "https://github.com/markswendsen-code/mcp-linkedin"
            ],
            env={
                **os.environ,
                "BROWSERBASE_API_KEY": os.getenv("BROWSERBASE_API_KEY", ""),
                "BROWSERBASE_CDP_URL": os.getenv("BROWSERBASE_CDP_URL", ""),
            }
        )

    async def search_jobs(self, keyword="Project Manager", location="Toronto"):
        if not os.getenv("BROWSERBASE_CDP_URL"):
            return (
                "LinkedIn MCP requires Browserbase setup.\n\n"
                "Missing environment variable: BROWSERBASE_CDP_URL\n\n"
                "Add this to your .env file:\n"
                "BROWSERBASE_CDP_URL=your_browserbase_cdp_url\n"
                "BROWSERBASE_API_KEY=your_browserbase_api_key"
            )

        try:
            async with stdio_client(self.server) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    tools = await session.list_tools()

                    if not tools.tools:
                        return "LinkedIn MCP connected, but no tools were found."

                    selected_tool = None

                    for tool in tools.tools:
                        tool_name = tool.name.lower()
                        tool_desc = (tool.description or "").lower()

                        if "job" in tool_name or "search" in tool_name or "job" in tool_desc:
                            selected_tool = tool
                            break

                    if selected_tool is None:
                        return "LinkedIn MCP connected, but no job-search tool was found."

                    arguments = self.build_arguments(
                        selected_tool.inputSchema,
                        keyword,
                        location
                    )

                    result = await session.call_tool(
                        selected_tool.name,
                        arguments
                    )

                    if not result.content:
                        return "LinkedIn MCP returned no job results."

                    return result.content[0].text

        except Exception as error:
            return (
                "LinkedIn MCP failed to connect or execute.\n\n"
                f"Reason: {error}\n\n"
                "Check that Browserbase is enabled and your .env values are correct."
            )

    def build_arguments(self, schema, keyword, location):
        properties = schema.get("properties", {})
        args = {}

        for field in properties:
            field_lower = field.lower()

            if field_lower in ["query", "keyword", "keywords", "search", "search_term"]:
                args[field] = keyword
            elif field_lower in ["location", "city", "place"]:
                args[field] = location
            elif field_lower in ["limit", "count", "max_results"]:
                args[field] = 10

        if not args:
            args = {
                "query": keyword,
                "location": location
            }

        return args


# ---------------- FINAL CHATBOT ----------------

class FinalJobAssistant:
    def __init__(self):
        self.market_service = MarketstackService()
        self.job_service = JobSemanticService()
        self.linkedin_service = LinkedInMCPService()

    async def handle_market_request(self, question):
        q = question.lower()

        if "tesla" in q:
            symbol = "TSLA"
        elif "microsoft" in q:
            symbol = "MSFT"
        elif "apple" in q:
            symbol = "AAPL"
        elif "amazon" in q:
            symbol = "AMZN"
        elif "google" in q or "alphabet" in q:
            symbol = "GOOGL"
        elif "nvidia" in q:
            symbol = "NVDA"
        else:
            symbol = "AAPL"

        return self.market_service.get_stock(symbol)

    async def handle_linkedin_request(self, question):
        q = question.lower()

        if "program manager" in q:
            keyword = "Program Manager"
        elif "technical project manager" in q:
            keyword = "Technical Project Manager"
        elif "project manager" in q:
            keyword = "Project Manager"
        else:
            keyword = "Project Manager"

        if "remote" in q:
            location = "Canada"
        elif "mississauga" in q:
            location = "Mississauga"
        elif "toronto" in q:
            location = "Toronto"
        else:
            location = "Toronto"

        return await self.linkedin_service.search_jobs(keyword, location)

    async def chat(self):
        self.job_service.build_vector_database()

        print("\nAI Job Assistant Ready")
        print("Service 1: Marketstack API")
        print("Service 2: Semantic Job Tracker")
        print("Service 3: LinkedIn MCP")
        print("Type 'exit' to quit.\n")

        while True:
            question = input("You: ").strip()

            if question.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break

            allowed, message = Guardrails.check(question)

            if not allowed:
                print(f"\nBot: {message}\n")
                continue

            q = question.lower()

            if "stock" in q or "market" in q or "share price" in q:
                  raw_answer = await self.handle_market_request(question)
                  answer = format_with_gpt(raw_answer, context="market")

            elif "linkedin" in q or "find jobs" in q or "search jobs" in q:
                answer = await self.handle_linkedin_request(question)

            else:
                answer = self.job_service.answer(question)

            # Keep this for other services
            if "stock" not in q:
              answer = format_with_gpt(answer)

            print(f"\nBot:\n{answer}\n")


if __name__ == "__main__":
    assistant = FinalJobAssistant()
    asyncio.run(assistant.chat())