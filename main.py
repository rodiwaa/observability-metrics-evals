from dotenv import load_dotenv
import os
from typing import Any, Literal, Sequence
from opik.integrations.langchain import OpikTracer
from queries import QUERIES, EXPECTED_ANSWERS
from opik.evaluation.metrics import (
    LLMJuriesJudge,
    Hallucination,
    ComplianceRiskJudge,
    DialogueHelpfulnessJudge,
    ContextPrecision,
    ContextRecall,
)
from opik import opik_context

print("setting env vars")
load_dotenv()

print(QUERIES)
print(QUERIES['HALLUCINATION'])

# print(os.getenv('OPENAI_API_KEY'))
print(f"LANGSMITH_TRACING: {os.getenv('LANGSMITH_TRACING')}")
print(f"LANGSMITH_PROJECT: {os.getenv('LANGSMITH_PROJECT')}")
print(f"OPIK_PROJECT_NAME: {os.getenv('OPIK_PROJECT_NAME')}")
print(f"OPIK_BASE_URL: {os.getenv('OPIK_BASE_URL')}")

# Instantiate Opik metrics once so every evaluation is tracked under the configured project.
context_precision_metric = ContextPrecision(project_name=os.getenv("OPIK_PROJECT_NAME"))
context_recall_metric = ContextRecall(project_name=os.getenv("OPIK_PROJECT_NAME"))
RAG_METRICS = {
    "context_precision": context_precision_metric,
    "context_recall": context_recall_metric,
}

from langchain_community.document_loaders import WebBaseLoader

urls = [
    "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
    "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
    "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
]

docs = [WebBaseLoader(url).load() for url in urls]

# print(docs[0][0].page_content.strip()[:1000])

from langchain_text_splitters import RecursiveCharacterTextSplitter

docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=100, chunk_overlap=50
)
doc_splits = text_splitter.split_documents(docs_list)

# print(doc_splits[0].page_content.strip())

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

vectorstore = InMemoryVectorStore.from_documents(
    documents=doc_splits, embedding=OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever()

from langchain_classic.tools.retriever import create_retriever_tool

retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_blog_posts",
    "Search and return information about Lilian Weng blog posts.",
)

# retriever_tool.invoke({"query": "types of reward hacking"})


from langgraph.graph import MessagesState
from langchain.chat_models import init_chat_model

response_model = init_chat_model("gpt-4o", temperature=0)


def generate_query_or_respond(state: MessagesState):
    """Call the model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply respond to the user.
    """
    response = (
        response_model
        .bind_tools([retriever_tool]).invoke(state["messages"])  
    )
    return {"messages": [response]}

# input = {"messages": [{"role": "user", "content": "hello!"}]}
# generate_query_or_respond(input)["messages"][-1].pretty_print()

input = {
    "messages": [
        {
            "role": "user",
            "content": "queries.",
        }
    ]
}
# print((input)["messages"])
# print((input)["messages"])[-1]

generate_query_or_respond(input)["messages"][-1].pretty_print()

from pydantic import BaseModel, Field

GRADE_PROMPT = (
    "You are a grader assessing relevance of a retrieved document to a user question. \n "
    "Here is the retrieved document: \n\n {context} \n\n"
    "Here is the user question: {question} \n"
    "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n"
    "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
)


class GradeDocuments(BaseModel):  
    """Grade documents using a binary score for relevance check."""

    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )


grader_model = init_chat_model("gpt-4o", temperature=0)


def grade_documents(
    state: MessagesState,
) -> Literal["generate_answer", "rewrite_question"]:
    """Determine whether the retrieved documents are relevant to the question."""
    question = state["messages"][0].content
    context = state["messages"][-1].content

    prompt = GRADE_PROMPT.format(question=question, context=context)
    response = (
        grader_model
        .with_structured_output(GradeDocuments).invoke(  
            [{"role": "user", "content": prompt}]
        )
    )
    score = response.binary_score

    if score == "yes":
        return "generate_answer"
    else:
        return "rewrite_question"
    
from langchain_core.messages import convert_to_messages, ToolMessage

input = {
    "messages": convert_to_messages(
        [
            {
                "role": "user",
                "content": "What does Lilian Weng say about types of reward hacking?",
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "1",
                        "name": "retrieve_blog_posts",
                        "args": {"query": "types of reward hacking"},
                    }
                ],
            },
            {"role": "tool", "content": "meow", "tool_call_id": "1"},
        ]
    )
}
grade_documents(input)


input = {
    "messages": convert_to_messages(
        [
            {
                "role": "user",
                "content": "What does Lilian Weng say about types of reward hacking?",
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "1",
                        "name": "retrieve_blog_posts",
                        "args": {"query": "types of reward hacking"},
                    }
                ],
            },
            {
                "role": "tool",
                "content": "reward hacking can be categorized into two types: environment or goal misspecification, and reward tampering",
                "tool_call_id": "1",
            },
        ]
    )
}
grade_documents(input)

from langchain.messages import HumanMessage

REWRITE_PROMPT = (
    "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
    "Here is the initial question:"
    "\n ------- \n"
    "{question}"
    "\n ------- \n"
    "Formulate an improved question:"
)


def rewrite_question(state: MessagesState):
    """Rewrite the original user question."""
    messages = state["messages"]
    question = messages[0].content
    prompt = REWRITE_PROMPT.format(question=question)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [HumanMessage(content=response.content)]}

input = {
    "messages": convert_to_messages(
        [
            {
                "role": "user",
                "content": "What does Lilian Weng say about types of reward hacking?",
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "1",
                        "name": "retrieve_blog_posts",
                        "args": {"query": "types of reward hacking"},
                    }
                ],
            },
            {"role": "tool", "content": "meow", "tool_call_id": "1"},
        ]
    )
}

response = rewrite_question(input)
print(response["messages"][-1].content)

GENERATE_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Use three sentences maximum and keep the answer concise.\n"
    "Question: {question} \n"
    "Context: {context}"
)


def generate_answer(state: MessagesState):
    """Generate an answer."""
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    
    # FIXME: is trace breaking my code?
    # TODO: issue is that get_current_trace_data will be None since this fn is not traced yet.
    # FIXME: call this function via langgraph node only, not outside LG so this LG node is traced properly.
    # update trace feedback metrics after #414 only. to capture all grapgh interactions w LLM.
    # move it all to a fn, duh
    trace = opik_context.get_current_trace_data()
    print(f"trace {trace}")

    jury = LLMJuriesJudge(
        judges=[
            Hallucination(model="gpt-4o-mini"),
            ComplianceRiskJudge(),
            DialogueHelpfulnessJudge(),
        ]
    )
    scores = jury.score(
        input=question,
        output=response.content,
    )

    # for name, metric in scores:
    #     print("inside metric jury")
    #     print(name, metric)

    # print(f"scores - ${scores}")
    # for score in score.items():
    #     print(score)
    #     trace.feedback_scores(
    #         {
    #             "category_name": "Test",
    #             "name": "Test",
    #             "reason": "Test",
    #             "value": 1,
    #         }
    #     )
    
    return {"messages": [response]}

input = {
    "messages": convert_to_messages(
        [
            {
                "role": "user",
                "content": "What does Lilian Weng say about types of reward hacking?",
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "1",
                        "name": "retrieve_blog_posts",
                        "args": {"query": "types of reward hacking"},
                    }
                ],
            },
            {
                "role": "tool",
                "content": "reward hacking can be categorized into two types: environment or goal misspecification, and reward tampering",
                "tool_call_id": "1",
            },
        ]
    )
}

response = generate_answer(input)
response["messages"][-1].pretty_print()

# # my own code
# print(f"SCORING TEST")
# jury = LLMJuriesJudge(
#     judges=[
#         Hallucination(model="gpt-4o-mini"),
#         ComplianceRiskJudge(),
#         DialogueHelpfulnessJudge(),
#     ]
# )

# score = jury.score(
#     input = QUERIES['REWARD_HACKING'],
#     output = response["messages"][-1].pretty_print()
# )

# # my own code till here...

# print(f"SCORE: ${score}")

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import ToolMessage

def retrieve(state: MessagesState):
    """Retrieve documents using the retriever tool."""
    last_message = state["messages"][-1]
    tool_calls = last_message.tool_calls
    tool_messages = []
    for tool_call in tool_calls:
        result = retriever_tool.invoke(tool_call["args"])
        tool_messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))
    return {"messages": tool_messages}

workflow = StateGraph(MessagesState)

# Define the nodes we will cycle between
workflow.add_node(generate_query_or_respond)
workflow.add_node("retrieve", retrieve)
workflow.add_node(rewrite_question)
workflow.add_node(generate_answer)

workflow.add_edge(START, "generate_query_or_respond")

# Decide whether to retrieve
workflow.add_conditional_edges(
    "generate_query_or_respond",
    # Assess LLM decision (call `retriever_tool` tool or respond to the user)
    tools_condition,
    {
        # Translate the condition outputs to nodes in our graph
        "tools": "retrieve",
        END: END,
    },
)

# Edges taken after the `action` node is called.
workflow.add_conditional_edges(
    "retrieve",
    # Assess agent decision
    grade_documents,
)
workflow.add_edge("generate_answer", END)
workflow.add_edge("rewrite_question", "generate_query_or_respond")

# Compile
graph = workflow.compile()

opik_tracer = OpikTracer(graph=graph.get_graph(xray=True))
opik_config = {
    "callbacks": [opik_tracer]
}

from IPython.display import Image, display

display(Image(graph.get_graph().draw_mermaid_png()))

for chunk in graph.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "What does Lilian Weng say about causes of hallucincation?",
            }
        ]
    },
config=opik_config
):
    for node, update in chunk.items():
        print("Update from node", node)
        update["messages"][-1].pretty_print()
        print("\n\n")
