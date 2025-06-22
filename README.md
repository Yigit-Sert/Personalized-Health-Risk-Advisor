# Personalized Health Insights & LLM Advisor

---

## 🚀 Running on Hugging Face Spaces

This project is fully compatible with [Hugging Face Spaces](https://huggingface.co/spaces). To deploy:

1. **Upload the entire repository to a new Space.**
2. **Set your API keys** (for LLM features) as [Hugging Face Secrets](https://huggingface.co/docs/hub/spaces-secrets) or enter them in the UI when prompted.
3. **No manual dataset download needed**—the app will fetch the dataset automatically.
4. **The app will launch automatically** using the provided `app.py` entry point.

---

This is a local Gradio application for training health prediction models, generating insights, and getting AI-driven advice.

## Prerequisites
- Python 3.9+
- A virtual environment tool (like `venv`)

## Setup Instructions

1.  **Clone the Repository:**
    ```bash
    git clone <your-repo-url>
    cd your_health_advisor_project
    ```

2.  **Create and Activate a Virtual Environment:**
    ```bash
    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the Dataset:**
    - Download the `heart_2022_no_nans.csv` file from the [Kaggle Dataset](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease).
    - Unzip the downloaded file. You might find the CSV inside a `2022` folder.
    - Place the `heart_2022_no_nans.csv` file directly inside the `data/` directory.

5.  **Configure API Keys & Local Models:**
    - Rename the `.env.example` file to `.env`.
    - Open the `.env` file and replace the placeholder values with your actual API keys for Google, Together AI, and Groq.
    - **For Local Models (Ollama):**
        - Ensure you have [Ollama](https://ollama.ai/) installed and running.
        - Pull the models you wish to use, for example: `ollama pull qwen:0.5b`.
        - Open `src/config.py` and update the `OLLAMA_MODELS` dictionary to match the models you have downloaded. The key is the display name for the UI, and the value is the exact model tag used by Ollama.

## Running the Application

Once the setup is complete, run the main application file from the root directory:

```bash
python -m src.app
```

The application will start, and you can access it at the local URL provided in your terminal (usually http://127.0.0.1:7860).

## Project Features

- **Local Model Storage**: Trained machine learning models are automatically saved to the `saved_models/` directory. On subsequent runs with the same data configuration, these models are loaded from disk, saving training time.

- **Multi-Provider LLM Integration**: Seamlessly switch between different LLM providers for health advice generation and evaluation, including:
    - Google Gemini
    - Together AI
    - Groq Cloud
    - **Local models via Ollama**, allowing for offline, private, and free inference using your own hardware.

- **Structured Code**: The project is separated into modules for UI (`app.py`), logic (`callbacks.py`), utilities (`utils.py`), and configuration (`config.py`).

- **Secure API Keys**: Cloud-based API keys are managed via a `.env` file and are not hardcoded. Local models run without needing any external keys. 