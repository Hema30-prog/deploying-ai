import asyncio
import gradio as gr

from final_job_assistant import FinalJobAssistant, Guardrails, format_with_gpt


assistant = FinalJobAssistant()


PERSONALITY = """
You are CareerPilot, a warm, practical, encouraging AI career assistant.
You help the user with job tracking, stock questions, and LinkedIn MCP job searches.
Be concise, supportive, and clear.
"""


async def respond(message, history):
    allowed, guardrail_message = Guardrails.check(message)

    if not allowed:
        return guardrail_message

    q = message.lower()

    try:
        if (
            "stock" in q
            or "market" in q
            or "share price" in q
            or "compare" in q
            or "top performing" in q
            or "performers" in q
        ):
            raw_answer = await assistant.handle_market_request(message)
            answer = format_with_gpt(
                f"{PERSONALITY}\n\nUser asked: {message}\n\nService result:\n{raw_answer}",
                context="market"
            )

        elif "linkedin" in q or "find jobs" in q or "search jobs" in q:
            raw_answer = await assistant.handle_linkedin_request(message)
            answer = format_with_gpt(
                f"{PERSONALITY}\n\nUser asked: {message}\n\nLinkedIn MCP result:\n{raw_answer}"
            )

        else:
            raw_answer = assistant.job_service.answer(message)
            answer = format_with_gpt(
                f"{PERSONALITY}\n\nUser asked: {message}\n\nJob tracker result:\n{raw_answer}"
            )

        return answer

    except Exception as error:
        return f"CareerPilot ran into an issue: {error}"


def chat_sync(message, history):
    return asyncio.run(respond(message, history))


if __name__ == "__main__":
    assistant.job_service.build_vector_database()

    demo = gr.ChatInterface(
        fn=chat_sync,
        title="CareerPilot AI Job Assistant",
        description=(
            "A friendly AI career assistant with 3 services: "
            "Marketstack API for worldwide stock market, Job Tracker Semantic Search using own dataset, and LinkedIn MCP."
        ),
        chatbot=gr.Chatbot(height=500, type="messages"),
        textbox=gr.Textbox(
            placeholder="Ask me: Which jobs did I apply for? Find Project Manager jobs on LinkedIn. Show Apple stock price.",
            container=False,
            scale=7,
        ),
        examples=[
            "Which jobs did I apply for?",
            "How many interviews did I get?",
            "Which site is more successful, LinkedIn or Indeed?",
            "Show me Apple stock price",
            "Compare Apple and Microsoft",
            "Find Project Manager jobs on LinkedIn in Toronto",
            "Tell me about cats",
        ],
    )

    demo.launch()