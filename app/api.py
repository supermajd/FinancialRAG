
"""api.py: FastAPI server for the H&M financial-document RAG chat endpoint."""

__author__ = "Majd Jamal"

import inspect
import json
import os
from contextlib import asynccontextmanager
from datetime import date

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel

from app.retrieve import Retrieval


def build_system_prompt() -> str:
    today = date.today().isoformat()
    return (
        f"You are a question-answering assistant for H&M Group's public corporate "
        f"documents — annual reports, AGM and EGM materials, corporate governance "
        f"reports, sustainability reports, and remuneration reports. Today is {today}. "

        f"HOW THIS WORKS: "
        f"When the user asks a question, call the search_documents tool. The tool returns "
        f"passages (chunks) from H&M's reports. These chunks are evidence. Use them to "
        f"answer the user's original question directly. "

        f"SEARCH QUERY RULES: "
        f"Write focused search queries that combine the metric or entity with the intent. "
        f"For example: "
        f"'what was the dividend in 2019?' -> 'dividend per share 2019 proposed by the board'. "
        f"'who sat on the board in 2017?' -> 'board of directors members 2017'. "
        f"Do not search only for 'H&M' if the user asks for a specific figure or fact. "

        f"SCOPE RULES: "
        f"Pass the year filter only when the user names an explicit year, and the category "
        f"filter only when they clearly mean one document type. "
        f"If the user says 'latest' or 'most recent', prefer the newest relevant chunk by year. "

        f"ANSWERING RULES: "
        f"Answer the user's specific question. Do not summarize all retrieved chunks. "
        f"First identify the chunk or chunks that best match the question. "
        f"Ignore unrelated chunks from other years, documents, or topics unless asked. "

        f"This corpus is about exact facts: report exact figures and exact wording from the "
        f"chunks — amounts (SEK million, %, per share), dates, names, board resolutions, and "
        f"targets. Quote the number or wording as it appears, and note the document and year. "
        f"The documents are in Swedish and English; Swedish financial terms matter: "
        f"'utdelning' = dividend, 'styrelse' = board, 'årsstämma' = annual general meeting (AGM), "
        f"'nettoomsättning' = net sales, 'rörelseresultat' = operating profit. "

        f"If the answer is visible in a chunk, give it directly with the figure. "
        f"Never invent or estimate a figure that is not present in the retrieved chunks. "
        f"Do not give investment advice or recommendations unless explicitly asked. "

        f"Be brief and factual: 1-3 sentences. "

        f"To search, you MUST invoke the search_documents tool through proper tool calling. "
        f"Do not write tool-call syntax as text in your reply. "

        f"If retrieval truly returns nothing relevant, say so honestly rather than guessing."
    )


class LocalLLM:
    """
    Local LLM Client

    Wraps a local Ollama server via the OpenAI-compatible chat completions API.
    Holds tool definitions and dispatches tool calls back to registered Python
    functions. Loops until the model produces a final text answer.
    """

    def __init__(self,
                 model: str = None,
                 base_url: str = None) -> None:

        self.model = model or os.environ.get(
            'OLLAMA_MODEL',
            'qwen2.5:14b-instruct')

        base_url = base_url or os.environ.get(
            'OLLAMA_BASE_URL',
            'http://localhost:11434/v1')

        self.client = OpenAI(base_url = base_url, api_key = 'ollama')
        self.tools = []
        self.dispatch = {}

    def register_tool(self,
                      name: str,
                      description: str,
                      parameters: dict,
                      fn) -> None:
        """ Registers a Python function as a tool the LLM can invoke.
        :param name: Tool name exposed to the LLM
        :param description: When the LLM should use this tool
        :param parameters: JSON schema for the tool's input
        :param fn: Python callable executed when the LLM selects this tool
        """

        self.tools.append({
            'type': 'function',
            'function': {
                'name':        name,
                'description': description,
                'parameters':  parameters,
            },
        })

        self.dispatch[name] = fn

    def ask(self, query: str, max_steps: int = 4) -> dict:
        """ Runs a chat turn, looping through tool calls until a text answer.
        :param query: User question
        :param max_steps: Maximum number of tool-call rounds before stopping
        :return result: Dict with 'answer' text and 'sources' list of chunks
        """

        conversation = [
            {'role': 'system', 'content': build_system_prompt()},
            {'role': 'user',   'content': query},
        ]

        sources = []

        for step in range(max_steps):

            response = self.client.chat.completions.create(
                model    = self.model,
                messages = conversation,
                temperature = 0.2,
                tools    = self.tools or None)

            message = response.choices[0].message

            # No tool call -> the model has produced the final answer.
            if not message.tool_calls:
                content = message.content or ''

                # Detect leaked tool-call syntax: model wrote the call as text instead of
                # invoking it. Retry the same step with a corrective hint.
                leaked = any(marker in content for marker in (
                    'FilterWhere', 'search_documents(', '"query":', 'tool_call',
                ))

                if leaked and step < max_steps - 1:
                    conversation.append({
                        'role':    'assistant',
                        'content': content,
                    })
                    conversation.append({
                        'role':    'user',
                        'content': (
                            'You wrote the tool call as text. Please invoke the '
                            'search_documents tool through the proper tool-calling mechanism, '
                            'not as text in your reply.'),
                    })
                    continue

                return {'answer': content, 'sources': sources}

            # Append the assistant message so the next round has full context.
            conversation.append({
                'role':       'assistant',
                'content':    message.content,
                'tool_calls': [tc.model_dump() for tc in message.tool_calls],
            })

            # Run each tool the model requested and feed results back.
            for tool_call in message.tool_calls:

                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments or '{}')

                if name in self.dispatch:
                    try:
                        # Strip kwargs the function doesn't accept.
                        fn = self.dispatch[name]
                        valid = set(inspect.signature(fn).parameters.keys())
                        clean_args = {k: v for k, v in args.items() if k in valid}
                        result = fn(**clean_args)
                    except Exception as e:
                        result = f'Tool error: {type(e).__name__}: {e}'
                else:
                    result = f'Unknown tool: {name}'

                # Capture retrieval results so the frontend can show citations.
                if name == 'search_documents' and isinstance(result, list):
                    sources.extend(result)

                conversation.append({
                    'role':         'tool',
                    'tool_call_id': tool_call.id,
                    'content':      json.dumps(result, default = str),
                })

        return {
            'answer':  'I could not produce a final answer within the tool budget.',
            'sources': sources,
        }


def search_documents(query: str,
                     k: int = 8,
                     year: int = None,
                     category: str = None) -> list:
    """ Tool that wraps Retrieval.search with flat LLM-friendly arguments.
    :return results: List of {chunk, document_title, category, year, source_url, score} dicts
    """

    print(f'search_documents: query = {query!r}, k = {k}, '
          f'year = {year}, category = {category}')

    filters = {}

    if year:
        filters['year'] = year
    if category:
        filters['category'] = category

    raw = app.state.retriever.search(
        text    = query,
        k       = k,
        filters = filters or None)

    results = []
    for rank, (chunk, meta, score) in enumerate(raw, start = 1):

        # Recover the clean chunk body (strip the Title/Type/Year/Section header).
        text = chunk.split('\n\n', 1)[-1]

        results.append({
            'rank':           rank,
            'score':          round(meta.get('similarity', 0.0), 4),
            'document_title': meta.get('document_title'),
            'type':           meta.get('meeting_type') or meta.get('category'),
            'year':           meta.get('year'),
            'heading':        meta.get('heading'),
            'text':           text,
        })

    return results


@asynccontextmanager
async def lifespan(app: FastAPI):

    print('\n =-=-=-=- Loading retrieval pipeline -=-=-=-= \n')

    app.state.retriever = Retrieval()
    app.state.llm = LocalLLM()

    app.state.llm.register_tool(
        name        = 'search_documents',
        description = (
            "Search H&M Group's corporate document archive (annual reports, AGM/EGM "
            "materials, corporate governance, sustainability, and remuneration reports) "
            "for passages matching a topic. Supports filtering by report year and "
            "document category. Use this whenever the user asks about a figure, board "
            "decision, resolution, policy, target, or any fact from H&M's reports."),
        parameters = {
            'type': 'object',
            'properties': {
                'query':    {'type': 'string',  'description': 'Topic, figure, or fact to search for.'},
                'year':     {'type': 'integer', 'description': 'Restrict to a specific report year, e.g. 2019.'},
                'category': {'type': 'string',  'description': (
                    "Restrict to a document type: 'annual_report', 'agm_egm', "
                    "'corporate_governance', 'sustainability_disclosure', or 'remuneration'.")},
            },
            'required': ['query'],
        },
        fn = search_documents)

    print('Ready')

    yield

    print('Shutting down')


app = FastAPI(lifespan = lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins = [
        "http://localhost:8080",
        "http://localhost:5173",
        "http://localhost:3000",
    ],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)


class ChatRequest(BaseModel):
    query: str


@app.get('/health')
async def health():
    return {'status': 'ok'}


@app.post('/chat')
async def chat(request: ChatRequest):
    """ RAG chat. The LLM decides whether to call search_documents and then
    synthesizes an answer grounded in the retrieved chunks.
    """

    result = app.state.llm.ask(request.query)

    return result