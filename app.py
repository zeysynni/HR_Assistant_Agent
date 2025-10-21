from assistant import Assistant
import gradio as gr

# Initialize the Assistant once at startup
HR = Assistant()
agent = HR.build_agent()

def chat(message, history):
    """Main chat handler for Gradio UI."""
    try:
        response = agent.invoke({"input": message})
        return response["output"]
    except Exception as e:
        return f"⚠️ An error occurred: {e}"

if __name__ == "__main__":
    with gr.Blocks() as view:
        gr.Markdown("## 💼 HR Assistant — Candidate Matcher")
        gr.Markdown(
            "You can paste a job URL directly in the chat — the assistant will use its scraper tool automatically."
        )

        gr.ChatInterface(
            fn=chat,
            type="messages",
            title="Chat with your HR Assistant"
        )

    view.launch(inbrowser=True)

