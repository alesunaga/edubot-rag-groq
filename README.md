🎓 EduBot RAG (Groq Edition)

A Didactically Aligned AI Tutor for K-12 Education

📖 Overview

EduBot RAG is an open-source educational chatbot designed to restore Teacher Agency in the AI era. Unlike generic AI assistants, EduBot uses Retrieval-Augmented Generation (RAG) to strictly constrain answers to teacher-uploaded materials (PDFs/Textbooks), preventing hallucinations and ensuring curricular alignment.

This "Groq Edition" is optimized for speed and sustainability:

Inference: Powered by Groq LPU (Llama 3) for near-instant responses.

Memory: Uses Local CPU Embeddings (sentence-transformers), eliminating data quota limits and costs for document processing.

🔬 Theoretical Framework

This project translates the research findings from the University of Konstanz into a functional prototype:

Regaining Control: The system enforces a "Strict Context Mode" to mitigate hallucinations (Pampel et al., 2025).

Didactic Alignment: The bot adopts specific pedagogical roles (Instructor, Partner, Assistant) defined by Lauber et al. (2025).

Systemic Integration: Addresses the "Will, Skill, Tool" framework by providing a low-code, privacy-first tool (Martin et al., 2025).

✨ Features

⚡ Ultra-Low Latency: Uses Groq API for real-time classroom interaction.

📚 Unlimited Local RAG: Process large textbooks locally without hitting API rate limits.

🎭 Didactic Personas:

Instructor: Scaffolds learning, avoids direct answers.

Partner: Checks progress and offers peer feedback.

Assistant: Simplifies language for accessibility.

🔒 Strict Safety Layer: Hard-coded system prompts prevent off-topic discussions.

📊 Analytics Dashboard: Tracks student engagement anonymously.

🚀 Installation

Prerequisites

Python 3.8 or higher.

A free Groq API Key.

Setup

Clone the repository:

git clone [https://github.com/alesunaga/edubot-rag-groq.git](https://github.com/alesunaga/edubot-rag-groq.git)
cd edubot-rag


Install dependencies:

pip install -r requirements.txt


Run the application:

python -m streamlit run rag_chatbot.py


🛠️ Usage Guide

Teacher Mode:

Enter your Groq API Key.

Upload your study material (PDF).

Select the Didactic Role (e.g., Learning Instructor).

Enable Strict Mode.

Student Mode:

Students interact with the bot in natural language.

The bot answers only based on the uploaded PDF.

📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

Developed by Alexsandro Sunaga, Educational Technology Coordinator.

Based on research by B. Pampel, S. Martin, and A.-M. Lauber (University of Konstanz).
