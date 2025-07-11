import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# === API KEYS ===
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Set environment variables for libraries that use them
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACE_API_KEY or ""
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY or ""
os.environ["TOGETHER_API_KEY"] = TOGETHER_API_KEY or ""
os.environ["GROQ_API_KEY"] = GROQ_API_KEY or ""


# === LLM Model Configurations ===
# Define model choices from different providers
# The key is the display name in the UI.
# The value is a tuple: (Provider/Library Name, model_identifier)

GOOGLE_GEMINI_MODELS = {
    "Gemini-Pro (Google)": ("Google Gemini", "gemini-pro"),
    "Gemini-1.5-Flash (Google)": ("Google Gemini", "gemini-1.5-flash-latest"),
}

TOGETHER_AI_MODELS = {
    "Mixtral-8x7B (Together AI)": ("Together AI", "mistralai/Mixtral-8x7B-Instruct-v0.1"),
}

GROQ_CLOUD_MODELS = {
    "Llama3-8B (Groq)": ("Groq Cloud", "llama3-8b-8192"),
    "Gemma2-9B (Groq)": ("Groq Cloud", "gemma2-9b-it"),
}

# --- OLLAMA LOCAL MODELS ---
# This dictionary will be populated by checking the local Ollama instance.
# Add user-friendly names for your local models here.
OLLAMA_MODELS = {
    "Qwen (Local)": "qwen:latest",
    "DeepSeek (Local)": "deepseek-r1:1.5b",
    "Qwen 0.5B (Local)": "qwen:0.5b",
}

# Convert Ollama models to the standard format for the main map
# The provider is marked as "Ollama" to be handled by a specific logic path.
OLLAMA_MODELS_FORMATTED = {
    display_name: ("Ollama", model_id)
    for display_name, model_id in OLLAMA_MODELS.items()
}

# Combine all model dictionaries into the main map for the UI
MODEL_CHOICES_MAP = {
    **GOOGLE_GEMINI_MODELS,
    **TOGETHER_AI_MODELS,
    **GROQ_CLOUD_MODELS,
    **OLLAMA_MODELS_FORMATTED,
}


# === GLOBAL VARIABLES & CONSTANTS ===
MODEL_SAVE_DIR = "saved_models"
DATA_DIR = "data"
DATASET_FILENAME = "heart_2022_no_nans.csv"
DATASET_PATH = os.path.join(DATA_DIR, DATASET_FILENAME)

# Mappings and constants
label_map_binary = {'Yes': 1, 'No': 0}
label_map_diabetes = {"No": 0, "No, pre-diabetes or borderline diabetes": 1, "Yes": 2, "Yes, but only during pregnancy (female)": 3}
label_map_diabetes_inv = {v: k for k, v in label_map_diabetes.items()}

MAX_PREDICTION_FORM_FIELDS = 150

SPECIFIC_INPUT_CONCEPTUAL_NAMES = {
    "state": "State", "age": "Age", "sex": "Sex", "bmi": "BMI",
    "general_health": "GeneralHealth", "physical_activities": "PhysicalActivities",
    "sleep_hours": "SleepHours", "smoked_100_cigs": "SmokedAtLeast100Cigarettes",
    "ecig_usage": "ECigaretteUsage", "race_ethnicity": "RaceEthnicityCategory",
    "deaf": "DeafOrHardOfHearing", "blind": "BlindOrVisionDifficulty",
    "difficulty_concentrating": "DifficultyConcentrating", "difficulty_walking": "DifficultyWalking",
    "difficulty_dressing": "DifficultyDressingBathing", "difficulty_errands": "DifficultyErrands",
    "chest_scan": "ChestScan", "hiv_testing": "HIVTesting", "flu_vax": "FluVaxLast12",
    "pneumo_vax": "PneumoVaxEver", "high_risk_last_year": "HighRiskLastYear", "covid_pos": "CovidPos",
}

PRESET_PROFILES = [
    "45-year-old male, smoker (10 cigarettes/day), BMI 28. Sedentary lifestyle. Often feels stressed.",
    "30-year-old female, non-smoker, exercises 3 times a week, BMI 22. Reports good general health but has a family history of heart disease.",
    "60-year-old male, diagnosed with hypertension, BMI 32. Diet includes many processed foods, rarely exercises. Complains of occasional chest tightness.",
    "25-year-old female, BMI 20. Social smoker (few cigarettes on weekends), vegetarian, reports high stress levels from work. Sleeps about 6 hours a night.",
    "50-year-old male, tries to be active (walks daily), BMI 26. Recently noticed increased thirst and more frequent urination. No known major health issues."
]

# Application CSS
APP_CSS = """
body { font-family: 'Arial', sans-serif; }
.gradio-container { max-width: 100% !important; padding: 10px !important; }
.gr-panel { border-radius: 8px !important; box-shadow: 0 4px 8px rgba(0,0,0,0.05) !important; padding: 15px !important; margin-bottom: 15px !important;}
.gr-button.gr-button-primary { background: linear-gradient(135deg, #007bff, #0056b3) !important; color: white !important; border: none !important; box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important; }
.gr-button.gr-button-primary:hover { background: linear-gradient(135deg, #0056b3, #004085) !important; }
.gr-markdown h1, .gr-markdown h2, .gr-markdown h3 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 5px; margin-top: 20px;}
.gr-markdown h1 { font-size: 1.8em; } .gr-markdown h2 { font-size: 1.5em; } .gr-markdown h3 { font-size: 1.3em; }
.status_box { padding: 12px 15px; border-radius: 6px; font-weight: 500; border-left-width: 5px; border-left-style: solid; margin-top: 8px; margin-bottom: 8px; }
.status_success { background-color: #e7f5ff !important; border-left-color: #2980b9 !important; color: #1d6093 !important; }
.status_warning { background-color: #fff9e6 !important; border-left-color: #f39c12 !important; color: #b5651d !important; }
.status_error { background-color: #ffeef0 !important; border-left-color: #e74c3c !important; color: #a93226 !important; }
.gr-accordion > .gr-block > .label { font-size: 1.15em !important; font-weight: bold !important; color: #2980b9 !important; }
#app_title { text-align: center; color: #3498db; font-size: 2.2em; margin-bottom: 20px; text-shadow: 1px 1px 2px #ecf0f1;}
.prediction_input_field { margin-bottom: 5px !important; } /* Generic textboxes */
.prediction-outcome { font-weight:bold; font-size:1.2em; color: #007bff; }
.progress-bar > div { background-color: #2980b9 !important; }
.progress-container .progress { background-color: #2980b9 !important; }
.progress-level div.fill { background-color: #2980b9 !important; background: #2980b9 !important; }
.progress-level .percent-text { color: white !important; font-weight: bold !important; }
.info-box { background-color: #f8f9fa !important; border: 1px solid #f8f9fa !important; border-radius: 6px; padding: 12px; margin-top: 8px; margin-bottom: 8px; }
.info-box-header { font-weight: bold; color: #2980b9; margin-bottom: 5px; font-size: 1.05em; }
.info-box .gradio-dataframe table, .info-box .gradio-dataframe table tr, .info-box .gradio-dataframe table tr:nth-child(even) { background-color: #f8f9fa !important; }
#training_log_md_id .gr-markdown { font-family: 'Courier New', Courier, monospace; font-size: 0.9em; white-space: pre-wrap; background-color: #f8f9fa !important; border: 1px solid #f8f9fa !important; border-radius: 6px; padding: 12px; max-height: 300px; overflow-y: auto; color: #333;}
#training_log_md_id h3 { color: #2980b9 !important; border-bottom: 1px solid #ddd !important; margin-top: 10px !important; font-size: 1.1em !important;}
""" 