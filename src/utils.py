import requests
import json
import io
import os
import zipfile
from io import BytesIO

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import markdown as md_lib
from sklearn.metrics import confusion_matrix
from together import Together
import google.generativeai as genai
from xhtml2pdf import pisa
import re
import together
from groq import Groq
import ollama

# Import configurations
from .config import (
    HUGGINGFACE_API_KEY, GOOGLE_API_KEY, TOGETHER_API_KEY, GROQ_API_KEY, MODEL_CHOICES_MAP
)

# Configure GenAI
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# === UNIVERSAL LLM CALL FUNCTION ===
def get_response(provider, model_id, prompt):
    if provider == "Hugging Face":
        headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
        api_url = f"https://api-inference.huggingface.co/models/{model_id}"
        payload = {"inputs": prompt, "parameters": {"max_new_tokens": 1024, "temperature": 0.7, "return_full_text": False}}
        try:
            r = requests.post(api_url, headers=headers, json=payload, timeout=60)
            r.raise_for_status()
            data = r.json()
            return data[0].get("generated_text", "") if isinstance(data, list) else data.get("generated_text", "")
        except requests.exceptions.RequestException as e: return f"Hugging Face API error: {str(e)}"
        except json.JSONDecodeError: return f"Hugging Face API error: Could not decode JSON response. Response text: {r.text}"
    elif provider == "Google Gemini":
        try:
            genai.configure(api_key=GOOGLE_API_KEY); gemini = genai.GenerativeModel(model_id)
            response = gemini.generate_content(prompt, generation_config=genai.types.GenerationConfig(max_output_tokens=1024, temperature=0.7))
            return response.text if response and hasattr(response, 'text') else "No valid response from Gemini."
        except Exception as e: return f"Google Gemini error: {str(e)}"
    elif provider == "Together AI":
        try:
            client = Together(api_key=TOGETHER_API_KEY)
            res = client.chat.completions.create(model=model_id, messages=[{"role": "user", "content": prompt}], max_tokens=1024, temperature=0.7)
            return res.choices[0].message.content if res and res.choices else "No response from Together AI."
        except Exception as e: return f"Together AI error: {str(e)}"
    elif provider == "Groq Cloud":
        messages = [{"role": "user", "content": prompt}]
        url = "https://api.groq.com/openai/v1/chat/completions"; headers = {"Content-Type": "application/json", "Authorization": f"Bearer {GROQ_API_KEY}"}
        payload = {"model": model_id, "messages": messages, "temperature": 0.7, "max_tokens": 1024}
        try:
            r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60); r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except requests.exceptions.HTTPError:
            try: error_detail = r.json().get('error', {}).get('message', r.text); return f"Groq Cloud HTTP error: {r.status_code} - {error_detail}"
            except json.JSONDecodeError: return f"Groq Cloud HTTP error: {r.status_code} - {r.text}"
        except Exception as e: return f"Groq Cloud error: {str(e)}"
    elif provider == "Ollama":
        messages = [{"role": "user", "content": prompt}]
        try:
            response = ollama.chat(model=model_id, messages=messages, stream=False)
            if 'message' in response and 'content' in response['message']:
                return response['message']['content']
            else:
                return f"ERROR: Invalid response format from Ollama for model '{model_id}'."
        except ollama.ResponseError as e:
            return f"ERROR: Ollama response error for model '{model_id}'. Is it installed? Details: {e.error}"
        except Exception as e:
            return f"ERROR: Could not connect to the Ollama server. Is it running? Details: {str(e)}"
    return "Unknown provider or API key not set."

# === DATA ANALYSIS & PLOTTING HELPERS ===
def parse_df_info(df_instance):
    s = io.StringIO(); df_instance.info(buf=s, verbose=True, show_counts=True)
    info_str = s.getvalue(); lines = info_str.split('\n'); columns_data = []; parsing_columns = False; memory_usage = ""; entry_count = ""
    for line in lines:
        if "RangeIndex:" in line or "Int64Index:" in line or "Index:" in line and "entries" in line: entry_count = line.strip(); break
    if not entry_count and len(lines) > 1: entry_count = lines[1]
    for line_idx, line in enumerate(lines):
        if line.strip() == "---" or (line.startswith(" # ") and "Column" in line and "Non-Null Count" in line and "Dtype" in line) :
            parsing_columns = True
            if line.startswith(" # "): continue
        if parsing_columns and (line.startswith("dtypes:") or line.strip() == ""):
            parsing_columns = False
            if line.startswith("dtypes:"):
                 memory_usage = line.split("memory usage: ")[-1].strip() if "memory usage: " in line else ""
                 if not memory_usage and line_idx + 1 < len(lines) and "memory usage:" in lines[line_idx+1]: memory_usage = lines[line_idx+1].split("memory usage: ")[-1].strip()
            break
        if parsing_columns and line.strip():
            parts = line.strip().split(maxsplit=3)
            if len(parts) == 4: col_name = parts[1]; non_null_info = parts[2]; dtype = parts[3]
            elif len(parts) == 3:
                col_name = parts[0]; non_null_info = parts[1]; dtype = parts[2]
                if not "non-null" in non_null_info : non_null_info = parts[1] + " " + parts[2]; dtype = line.strip().split(non_null_info)[-1].strip()
            else: continue
            columns_data.append({"Column": col_name, "Non-Null Count": non_null_info, "Dtype": dtype})
    info_df = pd.DataFrame(columns_data)
    return entry_count, info_df, memory_usage

def plot_target_distribution_generic(series, title, label_map_inv, plot_type="Bar Chart"):
    plt.close('all'); fig, ax = plt.subplots(figsize=(8, 5)); counts = series.value_counts().sort_index()
    if not isinstance(label_map_inv, dict): label_map_inv = {}
    tick_labels = [label_map_inv.get(i, str(i)) for i in counts.index]
    if plot_type == "Bar Chart": counts.plot(kind='bar', ax=ax); ax.set_xticklabels(tick_labels, rotation=45, ha="right"); ax.set_ylabel("Count")
    elif plot_type == "Pie Chart":
        if counts.empty: ax.text(0.5, 0.5, 'No data', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        else: ax.pie(counts, labels=tick_labels, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
    ax.set_title(title); ax.set_xlabel("Target Value" if plot_type == "Bar Chart" else ""); plt.tight_layout(); return fig

def plot_model_comparison(results_df):
    if results_df is None or results_df.empty: return None
    plt.close('all'); fig, ax = plt.subplots(figsize=(12, 7)); metric_cols = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    plot_df = results_df.set_index("Model")[metric_cols].copy()
    for col in metric_cols: plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce')
    plot_df.plot(kind='bar', ax=ax); ax.set_title("Model Performance Comparison"); ax.set_ylabel("Score")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right"); plt.legend(title="Metrics"); plt.tight_layout(); return fig

def plot_feature_importances(importances, feature_names, model_name, target_disease, top_n=15):
    if importances is None or feature_names is None or len(importances) == 0 or len(feature_names) == 0:
        fig, ax = plt.subplots(); ax.text(0.5, 0.5, 'No importances.', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes); return fig
    plt.close('all'); fig, ax = plt.subplots(figsize=(10, max(7, top_n * 0.4)))
    if isinstance(feature_names, pd.Index): feature_names = feature_names.tolist()
    importances_arr = np.array(importances)
    if importances_arr.ndim > 1:
        importances_arr = np.mean(importances_arr, axis=0)
    if len(importances_arr) != len(feature_names):
        ax.text(0.5, 0.5, f'Feature/Importance mismatch: {len(importances_arr)} importances, {len(feature_names)} features.', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes); return fig
    sorted_indices = np.argsort(importances_arr)[::-1]; top_n_actual = min(top_n, len(feature_names))
    top_indices = sorted_indices[:top_n_actual]
    ax.barh(range(top_n_actual), importances_arr[top_indices][::-1], align='center')
    ax.set_yticks(range(top_n_actual)); ax.set_yticklabels(np.array(feature_names)[top_indices][::-1])
    ax.set_xlabel("Importance"); ax.set_title(f"Top {top_n_actual} Importances ({model_name} for {target_disease})"); plt.tight_layout(); return fig

def plot_confusion_matrix_custom(y_true, y_pred, model_name, target_disease, labels_map):
    plt.close('all'); unique_numeric_labels = sorted(list(set(y_true) | set(y_pred)))
    display_labels = [labels_map.get(i, str(i)) for i in unique_numeric_labels]
    cm = confusion_matrix(y_true, y_pred, labels=unique_numeric_labels)
    fig, ax = plt.subplots(figsize=(max(6, len(unique_numeric_labels)*1.5), max(5, len(unique_numeric_labels)*1.2)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=display_labels, yticklabels=display_labels)
    ax.set_title(f'Confusion Matrix - {model_name} ({target_disease})'); ax.set_xlabel('Predicted'); ax.set_ylabel('True'); plt.tight_layout(); return fig

# === FILE SAVING HELPERS ===
def html_to_pdf(source_html_string, output_filename):
    result_file = open(output_filename, "w+b")
    pisa_status = pisa.CreatePDF(source_html_string.encode('utf-8'), dest=result_file, encoding='utf-8')
    result_file.close()
    return not pisa_status.err

def save_report_files(selected_formats, content_to_save_md):
    if not content_to_save_md or not content_to_save_md.strip():
        return None, "⚠️ No content available to save. Generate advice and/or evaluation first."
    if not selected_formats:
        return None, "⚠️ Please select at least one file format for saving."

    files_to_zip = []; base_filename = "health_insights_report"; temp_dir = "temp_downloads_health_app"; os.makedirs(temp_dir, exist_ok=True)

    try:
        if "Markdown (.md)" in selected_formats: files_to_zip.append((f"{base_filename}.md", content_to_save_md.encode('utf-8')))

        if "Plain Text (.txt)" in selected_formats:
            plain_text = content_to_save_md
            plain_text = re.sub(r'^##\s*(.*)$', r'\1', plain_text, flags=re.MULTILINE)
            plain_text = re.sub(r'^###\s*(.*)$', r'\1', plain_text, flags=re.MULTILINE)
            plain_text = plain_text.replace("**", "").replace("`", "")
            plain_text = re.sub(r'^\*\s+', '- ', plain_text, flags=re.MULTILINE)
            files_to_zip.append((f"{base_filename}.txt", plain_text.encode('utf-8')))

        html_content_for_pdf_str = ""
        if "HTML (.html)" in selected_formats or "PDF (.pdf)" in selected_formats:
            html_body = md_lib.markdown(content_to_save_md, extensions=['fenced_code', 'tables', 'nl2br', 'toc'])
            html_content_for_pdf_str = f"""<html><head><meta charset="UTF-8"><title>Health Insights Report</title>
            <style>
                     @font-face {{font-family: DejaVu Sans; src: url('https://kendo.cdn.telerik.com/2017.2.621/styles/fonts/DejaVu/DejaVuSans.ttf');}}
                     body {{ font-family: 'DejaVu Sans', Arial, sans-serif; line-height: 1.6; padding: 20px; color: #333; }}
                     h1, h2, h3 {{ color: #2c3e50; border-bottom: 1px solid #eaecef; padding-bottom: 0.3em; margin-top: 24px; margin-bottom: 16px; font-weight: 600;}}
                     h1 {{ font-size: 2em; }} h2 {{ font-size: 1.5em; }} h3 {{ font-size: 1.25em; }}
                     table {{ border-collapse: collapse; width: 100%; margin-bottom: 16px; border: 1px solid #dfe2e5; }}
                     th, td {{ border: 1px solid #dfe2e5; padding: 6px 13px; }}
                     th {{ background-color: #f6f8fa; font-weight: 600;}}
                     tr:nth-child(2n) {{ background-color: #f6f8fa; }}
                     pre {{ background-color: #f0f0f0; padding: 10px; border: 1px solid #ddd; border-radius:3px; overflow-x:auto; white-space: pre-wrap; word-wrap: break-word; font-family: 'Consolas', 'Courier New', monospace;}}
                     code {{ background-color: #f0f0f0; padding: .2em .4em; margin: 0; font-size: 85%; border-radius: 3px; font-family: 'Consolas', 'Courier New', monospace;}}
                     ul, ol {{ padding-left: 2em; margin-bottom: 16px;}}
                     li {{ margin-bottom: 0.25em; }}
                     hr {{ height: .25em; padding: 0; margin: 24px 0; background-color: #e1e4e8; border: 0; }}
            </style></head><body>{html_body}</body></html>"""
            if "HTML (.html)" in selected_formats: files_to_zip.append((f"{base_filename}.html", html_content_for_pdf_str.encode('utf-8')))

        if "PDF (.pdf)" in selected_formats:
            if not html_content_for_pdf_str:
                html_body_pdf_only = md_lib.markdown(content_to_save_md, extensions=['fenced_code', 'tables', 'nl2br', 'toc'])
                html_content_for_pdf_str = f"""<html><head><meta charset="UTF-8"><title>Health Insights Report</title>
                <style>
                         @font-face {{font-family: DejaVu Sans; src: url('https://kendo.cdn.telerik.com/2017.2.621/styles/fonts/DejaVu/DejaVuSans.ttf');}}
                         body {{ font-family: 'DejaVu Sans', Arial, sans-serif; line-height: 1.6; padding: 20px; color: #333; }}
                         h1, h2, h3 {{ color: #2c3e50; border-bottom: 1px solid #eaecef; padding-bottom: 0.3em; margin-top: 24px; margin-bottom: 16px; font-weight: 600;}}
                         table {{ border-collapse: collapse; width: 100%; margin-bottom: 16px; border: 1px solid #dfe2e5; }} th, td {{ border: 1px solid #dfe2e5; padding: 6px 13px; }}
                         pre {{ background-color: #f0f0f0; padding: 10px; border: 1px solid #ddd; border-radius:3px; overflow-x:auto; white-space: pre-wrap; word-wrap: break-word; font-family: 'Consolas', 'Courier New', monospace;}}
                </style></head><body>{html_body_pdf_only}</body></html>"""

            pdf_path = os.path.join(temp_dir, f"{base_filename}.pdf")
            pdf_generation_success = html_to_pdf(html_content_for_pdf_str, pdf_path)
            if pdf_generation_success:
                with open(pdf_path, "rb") as f_pdf: files_to_zip.append((f"{base_filename}.pdf", f_pdf.read()))
                try: os.remove(pdf_path)
                except OSError as e_os: print(f"Note: Error removing temporary PDF file {pdf_path}: {e_os}")
            else:
                error_msg_pdf = "Error: PDF conversion failed. The HTML content might be too complex or contain unsupported elements. A TXT version of this error message is included."
                files_to_zip.append((f"{base_filename}_pdf_conversion_error.txt", error_msg_pdf.encode('utf-8')))
                print(f"PDF generation failed for {base_filename}.pdf")

        if not files_to_zip:
            return None, "⚠️ No files were generated based on selected formats or content."

        if len(files_to_zip) == 1:
            filename_single, content_bytes_single = files_to_zip[0]
            temp_file_path_single = os.path.join(temp_dir, filename_single)
            with open(temp_file_path_single, "wb") as f_single: f_single.write(content_bytes_single)
            return temp_file_path_single, f"✅ Report saved as `{filename_single}`. Download link should appear below."
        else:
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for filename_zip, content_bytes_zip in files_to_zip: zf.writestr(filename_zip, content_bytes_zip)
            zip_filename = f"{base_filename}_reports.zip"; zip_file_path = os.path.join(temp_dir, zip_filename)
            with open(zip_file_path, "wb") as f_zip: f_zip.write(zip_buffer.getvalue())
            return zip_file_path, f"✅ Reports zipped as `{zip_filename}`. Download link should appear below."

    except Exception as e_save_main:
        import traceback
        tb_str = traceback.format_exc()
        return None, f"❌ Error during file saving process: {str(e_save_main)}\nTraceback: {tb_str}" 