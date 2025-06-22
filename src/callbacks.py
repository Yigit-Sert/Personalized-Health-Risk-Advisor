import os
import gradio as gr
import pandas as pd
import warnings
import json
import re
import time
import joblib
import numpy as np
import traceback
import kagglehub

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

from config import *
from utils import *

warnings.filterwarnings("ignore")

# === APPLICATION STATE (managed as global variables in this module) ===
df = None; df_encoded = None
X_train, X_test, y_train, y_test = None, None, None, None
tree = None; rules = None
generated_advice_text = ""; advice_evaluation_text = ""
all_trained_models = {}
current_target_disease = "HadDiabetes"
available_disease_targets = []
model_feature_names = []
original_feature_names_for_form = []
form_creation_helper_data = {}

# === PREDICTION FORM HELPERS ===
age_categories_map = { (0, 18): "AgeCategory1824", (18, 25): "AgeCategory1824", (25, 30): "AgeCategory2529", (30, 35): "AgeCategory3034", (35, 40): "AgeCategory3539", (40, 45): "AgeCategory4044", (45, 50): "AgeCategory4549", (50, 55): "AgeCategory5054", (55, 60): "AgeCategory5559", (60, 65): "AgeCategory6064", (65, 70): "AgeCategory6569", (70, 75): "AgeCategory7074", (75, 80): "AgeCategory7579", (80, 200): "AgeCategory80orolder" }

def get_age_category_feature(age, feature_list):
    if not feature_list: return None
    cleaned_model_feature_list = [f.lower() for f in feature_list]
    for age_range, cat_name_base in age_categories_map.items():
        if age_range[0] <= age < age_range[1]:
            potential_feature_name_cleaned = cat_name_base.lower()
            if potential_feature_name_cleaned in cleaned_model_feature_list:
                original_idx = cleaned_model_feature_list.index(potential_feature_name_cleaned)
                return feature_list[original_idx]
    return None

def parse_preset_profile_for_form(profile_text):
    parsed_values = {}; age_match = re.search(r"(\d+)[-\s]?year[s]?-old", profile_text, re.IGNORECASE)
    if age_match: parsed_values['age'] = int(age_match.group(1))
    bmi_match = re.search(r"BMI\s*(\d+\.?\d*)", profile_text, re.IGNORECASE)
    if bmi_match: parsed_values['bmi'] = float(bmi_match.group(1))
    if "smoker" in profile_text.lower() and "non-smoker" not in profile_text.lower() and "not a smoker" not in profile_text.lower(): parsed_values['smoked_100_cigs'] = "Yes"
    elif "non-smoker" in profile_text.lower() or "not a smoker" in profile_text.lower(): parsed_values['smoked_100_cigs'] = "No"
    if "male" in profile_text.lower(): parsed_values['sex'] = "Male"
    elif "female" in profile_text.lower(): parsed_values['sex'] = "Female"
    return parsed_values

# === TAB-SPECIFIC FUNCTIONS (CALLBACKS) ===

def load_dataset_tab():
    global df, available_disease_targets
    available_disease_targets = []
    try:
        path = kagglehub.dataset_download("kamilpytlak/personal-key-indicators-of-heart-disease"); dataset_dir = path; csv_file_path = ""
        potential_paths = [ os.path.join(dataset_dir, "2022", "heart_2022_no_nans.csv"), os.path.join(dataset_dir, "heart_2022_no_nans.csv")]
        for p_path in potential_paths:
            if os.path.exists(p_path): csv_file_path = p_path; break
        if not csv_file_path:
            for root, _, files in os.walk(dataset_dir):
                for file_name in files:
                    if "heart_2022_no_nans.csv" in file_name: csv_file_path = os.path.join(root, file_name); break
                if csv_file_path: break
        if not csv_file_path or not os.path.exists(csv_file_path): return (gr.Markdown("❌ Error: `heart_2022_no_nans.csv` not found.", elem_classes="status_box status_error"), gr.update(visible=False), gr.update(value=None), gr.update(value=""), gr.update(value=None), gr.update(value=""), gr.update(selected="load_tab"), gr.update(choices=[], value=None))
        df = pd.read_csv(csv_file_path); num_rows, num_cols = df.shape
        shape_md = f"**Dataset Dimensions:**\n- Rows: {num_rows}\n- Columns: {num_cols}"
        entry_count_str, info_df_data, memory_usage_str = parse_df_info(df)
        info_summary_md = f"**Dataframe Structure Overview:**\n- {entry_count_str}\n- Memory Usage: {memory_usage_str}"
        status_md = gr.Markdown("✅ Dataset loaded successfully.", elem_classes="status_box status_success")
        return (status_md, gr.update(visible=True), gr.update(value=df.head()), gr.update(value=shape_md), gr.update(value=info_df_data), gr.update(value=info_summary_md), gr.update(selected="clean_tab"), gr.update(choices=[], value=None))
    except Exception as e: df = None; error_md = gr.Markdown(f"❌ Error loading: {str(e)}", elem_classes="status_box status_error"); return (error_md, gr.update(visible=False), gr.update(value=None), gr.update(value=""), gr.update(value=None), gr.update(value=""), gr.update(selected="load_tab"), gr.update(choices=[], value=None))

def clean_encode_tab():
    global df, df_encoded, available_disease_targets, model_feature_names, original_feature_names_for_form, form_creation_helper_data
    if df is None: status_md = gr.Markdown("⚠️ Load dataset first.", elem_classes="status_box status_warning"); return (status_md, gr.update(visible=False), gr.update(value=None), gr.update(value=""), gr.update(value=None), gr.update(value=""), "", gr.update(choices=[], value=None), gr.update(selected="load_tab"))
    try:
        df_temp = df.copy()
        original_cols_from_df = df.columns.tolist()
        cleaned_column_names = [re.sub(r'[-_ ]', '', col.strip()) for col in df_temp.columns]
        df_temp.columns = cleaned_column_names
        potential_target_cols = [col for col in df_temp.columns if col.startswith("Had")]; processed_targets = []
        for col in potential_target_cols:
            if df_temp[col].dtype == 'object':
                unique_vals = df_temp[col].unique()
                if col == "HadDiabetes": df_temp[col] = df_temp[col].map(label_map_diabetes).fillna(0); processed_targets.append(col)
                elif all(val in ['Yes', 'No'] for val in unique_vals if pd.notna(val)): df_temp[col] = df_temp[col].map(label_map_binary).fillna(0).astype(int); processed_targets.append(col)
        current_available_targets = list(processed_targets); had_diabetes_col_cleaned = "HadDiabetes"
        if had_diabetes_col_cleaned in df_temp.columns and had_diabetes_col_cleaned not in current_available_targets:
            if df_temp[had_diabetes_col_cleaned].dtype == 'object': df_temp[had_diabetes_col_cleaned] = df_temp[had_diabetes_col_cleaned].map(label_map_diabetes).fillna(0); current_available_targets.insert(0, had_diabetes_col_cleaned)
        available_disease_targets = current_available_targets if current_available_targets else []
        if not available_disease_targets and had_diabetes_col_cleaned in df_temp.columns: available_disease_targets = [had_diabetes_col_cleaned]
        original_feature_names_for_form = [col for col in df_temp.columns if col not in available_disease_targets]
        cleaned_to_original_map = {cleaned: original for cleaned, original in zip(df_temp.columns, original_cols_from_df)}
        form_creation_helper_data = {}
        for col_name_cleaned in original_feature_names_for_form:
            original_col_name_for_df_access = cleaned_to_original_map.get(col_name_cleaned, col_name_cleaned)
            current_col_data_temp = df_temp[col_name_cleaned]
            is_categorical_guess = False
            if current_col_data_temp.dtype == 'object': is_categorical_guess = True
            elif current_col_data_temp.nunique() < 35 and current_col_data_temp.nunique() > 0 :
                if not (pd.api.types.is_numeric_dtype(current_col_data_temp.dtype) and current_col_data_temp.nunique() > 10 ):
                     is_categorical_guess = True
            if is_categorical_guess:
                unique_vals = sorted(list(set(str(x).strip() for x in df[original_col_name_for_df_access].unique() if pd.notna(x))))
                form_creation_helper_data[col_name_cleaned] = {"type": "categorical", "original_cleaned_name": col_name_cleaned, "options": unique_vals}
            elif pd.api.types.is_numeric_dtype(current_col_data_temp.dtype):
                 form_creation_helper_data[col_name_cleaned] = {
                    "type": "numerical", "original_cleaned_name": col_name_cleaned,
                    "mean": current_col_data_temp.mean() if not current_col_data_temp.empty else 0,
                    "std": current_col_data_temp.std() if not current_col_data_temp.empty else 0
                }
            elif "agecategory" in col_name_cleaned.lower():
                form_creation_helper_data[col_name_cleaned] = {"type": "pre_encoded_categorical", "original_cleaned_name": col_name_cleaned}
        categorical_cols_to_encode = [ item["original_cleaned_name"] for item in form_creation_helper_data.values() if item["type"] == "categorical" ]
        df_encoded = pd.get_dummies(df_temp, columns=categorical_cols_to_encode, drop_first=True, dummy_na=False)
        newly_encoded_cols_names = list(set(df_encoded.columns) - set(df_temp.columns))
        num_rows_enc, num_cols_enc = df_encoded.shape; shape_md_enc = f"**Encoded Dataset Dimensions:**\n- Rows: {num_rows_enc}\n- Columns: {num_cols_enc}"
        entry_count_str_enc, info_df_enc_data, memory_usage_str_enc = parse_df_info(df_encoded)
        info_summary_md_enc = f"**Encoded Structure Overview:**\n- {entry_count_str_enc}\n- Memory: {memory_usage_str_enc}"
        change_summary_md = (f"**Transformations:** Initial: {len(original_cols_from_df)} cols. Cleaned: {len(df_temp.columns)}. Encoded: {len(df_encoded.columns)}.\n"
                             f"Targets Mapped: `{', '.join(processed_targets if processed_targets else ['None'])}`.\n"
                             f"One-Hot Encoded (Conceptual): `{', '.join(categorical_cols_to_encode) if categorical_cols_to_encode else 'None'}`.\n"
                             f"New One-Hot Columns (Sample): `{', '.join(newly_encoded_cols_names[:3])}...` (Total: {len(newly_encoded_cols_names)})")
        default_target_for_split_tab = "HadDiabetes" if "HadDiabetes" in available_disease_targets else available_disease_targets[0] if available_disease_targets else None
        if not available_disease_targets:
            status_md = gr.Markdown("⚠️ No target variables found/processed.", elem_classes="status_box status_warning")
            return (status_md, gr.update(visible=False), gr.update(value=None), gr.update(value=""), gr.update(value=None), gr.update(value=""), "", gr.update(choices=[], value=None), gr.update(selected="clean_tab"))
        status_md = gr.Markdown("✅ Data cleaning, mapping, and encoding completed.", elem_classes="status_box status_success")
        return (status_md, gr.update(visible=True), gr.update(value=df_encoded.head()), gr.update(value=shape_md_enc), gr.update(value=info_df_enc_data), gr.update(value=info_summary_md_enc), gr.update(value=change_summary_md),
                gr.update(choices=available_disease_targets, value=default_target_for_split_tab),
                gr.update(selected="split_tab"))
    except Exception as e:
        df_encoded = None; available_disease_targets = []; model_feature_names = []; original_feature_names_for_form = []; form_creation_helper_data = {}
        error_md = gr.Markdown(f"❌ Clean/encode error: {str(e)}", elem_classes="status_box status_error")
        return (error_md, gr.update(visible=False), gr.update(value=None), gr.update(value=""), gr.update(value=None), gr.update(value=""), "", gr.update(choices=[], value=None), gr.update(selected="clean_tab"))

def split_data_tab(target_disease_to_train, y_train_dist_type, y_test_dist_type):
    global df_encoded, X_train, X_test, y_train, y_test, current_target_disease, model_feature_names
    if df_encoded is None:
        status_md = gr.Markdown("⚠️ Clean/encode data first (go to Tab 2).", elem_classes="status_box status_warning")
        return (status_md, "", "", None, None, None, None, None, None, "", gr.update(visible=False), gr.update(selected="clean_tab"))
    current_target_disease = target_disease_to_train
    if not current_target_disease:
        status_md = gr.Markdown("⚠️ Please select a target disease from the dropdown above.", elem_classes="status_box status_warning")
        return (status_md, "", "", None, None, None, None, None, None, "", gr.update(visible=False), gr.update(selected="split_tab"))
    if current_target_disease not in df_encoded.columns:
        status_md = gr.Markdown(f"❌ Target '{current_target_disease}' not found in encoded data. Please re-run 'Clean & Encode'.", elem_classes="status_box status_error")
        return (status_md, "", "", None, None, None, None, None, None, "", gr.update(visible=False), gr.update(selected="clean_tab"))
    try:
        y = df_encoded[current_target_disease]; cols_to_drop_from_X = [col for col in df_encoded.columns if col.startswith("Had")]; X = df_encoded.drop(columns=cols_to_drop_from_X)
        model_feature_names = list(X.columns)
        stratify_option = y if y.nunique() > 1 else None; X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify_option)
        shapes_info_md = (f"**Split Info (Target: `{current_target_disease}`):**\n- X_train: {X_train.shape}, X_test: {X_test.shape}\n- y_train: {y_train.shape}, y_test: {y_test.shape}")
        inv_map = label_map_diabetes_inv if current_target_disease == "HadDiabetes" else {0: "No", 1: "Yes"}
        y_train_counts = y_train.value_counts().sort_index(); y_test_counts = y_test.value_counts().sort_index()
        fig_train_dist = plot_target_distribution_generic(y_train, f"Train Target: {current_target_disease}", inv_map, y_train_dist_type)
        fig_test_dist = plot_target_distribution_generic(y_test, f"Test Target: {current_target_disease}", inv_map, y_test_dist_type)
        train_dist_table = pd.DataFrame({'Label': [inv_map.get(k, str(k)) for k in y_train_counts.index], 'Count': y_train_counts.values})
        test_dist_table = pd.DataFrame({'Label': [inv_map.get(k, str(k)) for k in y_test_counts.index], 'Count': y_test_counts.values})
        status_md = gr.Markdown(f"✅ Data split for target: `{current_target_disease}`.", elem_classes="status_box status_success")
        return (status_md, shapes_info_md, f"**Target:** `{current_target_disease}`", gr.update(value=fig_train_dist if y_train_dist_type != "Table" else None, visible=y_train_dist_type != "Table"), gr.update(value=train_dist_table if y_train_dist_type == "Table" else None, visible=y_train_dist_type == "Table"), gr.update(value=fig_test_dist if y_test_dist_type != "Table" else None, visible=y_test_dist_type != "Table"), gr.update(value=test_dist_table if y_test_dist_type == "Table" else None, visible=y_test_dist_type == "Table"), y_train_counts, y_test_counts, json.dumps(inv_map), gr.update(visible=True), gr.update(selected="train_tab"))
    except Exception as e:
        X_train, X_test, y_train, y_test = [None]*4; model_feature_names = []
        error_md = gr.Markdown(f"❌ Error splitting: {str(e)}", elem_classes="status_box status_error")
        return (error_md, "", "", None, None, None, None, None, None, "", gr.update(visible=False), gr.update(selected="split_tab"))

def update_target_viz(counts_series_obj, target_disease, inv_map_json_str, viz_type, viz_title_prefix):
    if counts_series_obj is None or not inv_map_json_str : return gr.update(value=None, visible=False), gr.update(value=None, visible=False)
    try: counts_series = counts_series_obj; inv_map = json.loads(inv_map_json_str)
    except (json.JSONDecodeError, TypeError): return gr.update(value=None, visible=False), gr.update(value=None, visible=False)
    title = f"{viz_title_prefix} Target: {target_disease}"
    if viz_type == "Table":
        if counts_series.empty: table_data = pd.DataFrame(columns=['Label', 'Count'])
        else: table_data = pd.DataFrame({'Label': [inv_map.get(k, str(k)) for k in counts_series.index], 'Count': counts_series.values})
        return gr.update(value=None, visible=False), gr.update(value=table_data, visible=True)
    else: plot = plot_target_distribution_generic(counts_series, title, inv_map, viz_type); return gr.update(value=plot, visible=True), gr.update(value=None, visible=False)

def train_model_tab(progress=gr.Progress(track_tqdm=True)):
    global tree, X_train, y_train, X_test, y_test, all_trained_models, current_target_disease, model_feature_names, MODEL_SAVE_DIR
    training_log = ["### Model Training Log", f"Target Disease: **{current_target_disease}**"]
    yield (gr.Markdown("⏳ Starting model training process..."), None, "", None, None, None, None,
           gr.update(choices=[],value=None), gr.update(choices=[],value=None), None,
           gr.update(visible=False), gr.update(selected="train_tab"), "\n".join(training_log))

    if X_train is None or y_train is None or X_test is None or y_test is None:
        training_log.append("❌ Error: Training/testing data not available. Split data first (Tab 3).")
        yield (gr.Markdown("⚠️ Split data first (Tab 3).", elem_classes="status_box status_warning"), None, "", None, None, None, None,
               gr.update(choices=[],value=None), gr.update(choices=[],value=None), None,
               gr.update(visible=False), gr.update(selected="split_tab"), "\n".join(training_log))
        return

    try:
        os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
        all_trained_models.clear()
        models_config = {
            "Decision Tree": DecisionTreeClassifier(max_depth=7, random_state=42, min_samples_leaf=10),
            "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000, solver='liblinear'),
            "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42, n_estimators=100, max_depth=5),
            "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7)
        }
        results = []
        dt_fi, rf_fi, gb_fi = None, None, None
        model_preds = {}
        inv_map = label_map_diabetes_inv if current_target_disease == "HadDiabetes" else {0: "No", 1: "Yes"}
        current_X_train_features = X_train.columns.tolist()
        total_models = len(models_config)
        for i, (name, model_prototype) in enumerate(models_config.items()):
            model_instance_for_run = None
            progress((i + 0.1) / total_models, desc=f"Preparing {name} for {current_target_disease}...")
            training_log.append(f"\n⏳ Preparing '{name}' for target '{current_target_disease}'...")
            yield (gr.update(), None, "", None, None, None, None, gr.update(), gr.update(), None, gr.update(), gr.update(), "\n".join(training_log))
            safe_model_name_part = re.sub(r'\W+', '', name)
            safe_target_name_part = re.sub(r'\W+', '', current_target_disease)
            model_filename = f"{safe_model_name_part}_{safe_target_name_part}.joblib"
            features_filename = f"{safe_model_name_part}_{safe_target_name_part}_features.json"
            model_path = os.path.join(MODEL_SAVE_DIR, model_filename)
            features_path = os.path.join(MODEL_SAVE_DIR, features_filename)
            loaded_successfully_from_disk = False
            if os.path.exists(model_path) and os.path.exists(features_path):
                training_log.append(f"  Found saved files for '{name}'. Attempting to load...")
                time.sleep(0.1)
                yield (gr.update(), None, "", None, None, None, None, gr.update(), gr.update(), None, gr.update(), gr.update(), "\n".join(training_log))
                try:
                    candidate_model = joblib.load(model_path)
                    with open(features_path, 'r') as f:
                        saved_features = json.load(f)
                    if saved_features == current_X_train_features:
                        model_instance_for_run = candidate_model
                        all_trained_models[name] = model_instance_for_run
                        training_log.append(f"  ✅ Loaded pre-trained model '{name}' from disk. Features match.")
                        loaded_successfully_from_disk = True
                    else:
                        training_log.append(f"  ⚠️ Feature set mismatch for '{name}'. Saved model features (count: {len(saved_features)}) differ from current data features (count: {len(current_X_train_features)}). Retraining necessary.")
                        if len(saved_features) == len(current_X_train_features):
                            diff_saved = [f for f in saved_features if f not in current_X_train_features][:3]
                            diff_current = [f for f in current_X_train_features if f not in saved_features][:3]
                            if diff_saved or diff_current:
                                training_log.append(f"     Example differing features - Saved: {diff_saved}, Current: {diff_current}")
                except Exception as e_load:
                    training_log.append(f"  ❌ Error loading model '{name}' from disk: {str(e_load)}. Will retrain.")
            if not loaded_successfully_from_disk:
                training_log.append(f"  🚀 Training new model '{name}' (or retrying due to previous issue/mismatch).")
                time.sleep(0.1)
                yield (gr.update(), None, "", None, None, None, None, gr.update(), gr.update(), None, gr.update(), gr.update(), "\n".join(training_log))
                model_instance_for_run = model_prototype
                model_instance_for_run.fit(X_train, y_train)
                all_trained_models[name] = model_instance_for_run
                try:
                    joblib.dump(model_instance_for_run, model_path)
                    with open(features_path, 'w') as f:
                        json.dump(current_X_train_features, f)
                    training_log.append(f"  💾 Saved (re)trained model '{name}' and its features to disk.")
                except Exception as e_save:
                    training_log.append(f"  ❌ Error saving (re)trained model '{name}': {str(e_save)}. Model is in memory for this session.")
            progress((i + 0.5) / total_models, desc=f"Evaluating {name}...")
            training_log.append(f"🔬 Evaluating {name} (whether loaded or newly trained)...")
            yield (gr.update(), None, "", None, None, None, None, gr.update(), gr.update(), None, gr.update(), gr.update(), "\n".join(training_log))
            y_pred = model_instance_for_run.predict(X_test)
            model_preds[name] = y_pred
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            results.append({"Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1-Score": f1})
            if loaded_successfully_from_disk:
                training_log.append(f"  👍 Evaluation complete for pre-loaded '{name}'. Accuracy: {acc:.4f}, F1: {f1:.4f}")
            else:
                training_log.append(f"  ✅ '{name}' (re)trained & evaluated. Accuracy: {acc:.4f}, F1: {f1:.4f}")
            if name == "Decision Tree":
                tree = model_instance_for_run
                dt_fi = getattr(tree, 'feature_importances_', None)
            elif name == "Random Forest":
                rf_fi = getattr(model_instance_for_run, 'feature_importances_', None)
            elif name == "Gradient Boosting":
                gb_fi = getattr(model_instance_for_run, 'feature_importances_', None)
            progress((i + 1) / total_models, desc=f"{name} Complete.")
            yield (gr.update(), None, "", None, None, None, None, gr.update(), gr.update(), None, gr.update(), gr.update(), "\n".join(training_log))
        rdf = pd.DataFrame(results); metrics = ['Accuracy','Precision','Recall','F1-Score']; rdf[metrics]=rdf[metrics].apply(pd.to_numeric)
        sdf = rdf.style.highlight_max(axis=0, subset=metrics, props='background-color:lightgreen; font-weight:bold;').format(formatter="{:.4f}", subset=metrics)
        best_sum = f"### Best Model per Metric (`{current_target_disease}`):\n" + "".join([f"- **{m}:** {rdf.loc[rdf[m].idxmax(),'Model']} ({rdf[m].max():.4f})\n" for m in metrics])
        fig_comp = plot_model_comparison(rdf)
        fig_dt=plot_feature_importances(dt_fi, current_X_train_features, "DT", current_target_disease) if dt_fi is not None else None
        fig_rf=plot_feature_importances(rf_fi, current_X_train_features, "RF", current_target_disease) if rf_fi is not None else None
        fig_gb=plot_feature_importances(gb_fi, current_X_train_features, "GB", current_target_disease) if gb_fi is not None else None
        model_names_for_dd = list(all_trained_models.keys()); next_tab = "rules_tab" if tree else "predict_tab"
        cm_model_default = "Decision Tree" if "Decision Tree" in model_names_for_dd else model_names_for_dd[0] if model_names_for_dd else None
        fig_cm = None
        if cm_model_default and y_test is not None and cm_model_default in model_preds:
             fig_cm = plot_confusion_matrix_custom(y_test, model_preds[cm_model_default], cm_model_default, current_target_disease, inv_map)
        else:
            training_log.append(f"⚠️ Could not generate CM for {cm_model_default} (y_test or predictions missing).")
        training_log.append("\n🎉 All models processed (loaded or trained) and evaluated successfully! Results shown are for these active models.")
        status_md_final = gr.Markdown(f"✅ Models ready for `{current_target_disease}`. Evaluation metrics displayed are for the models currently active (loaded or re-trained).", elem_classes="status_box status_success")
        yield (status_md_final, sdf, best_sum, fig_comp, fig_dt, fig_rf, fig_gb,
               gr.update(choices=model_names_for_dd,value=model_names_for_dd[0] if model_names_for_dd else None),
               gr.update(choices=model_names_for_dd,value=cm_model_default),
               fig_cm, gr.update(visible=True), gr.update(selected=next_tab), "\n".join(training_log))
    except Exception as e:
        tree=None; all_trained_models.clear()
        training_log.append(f"\n❌ Major error during training/loading: {str(e)}")
        training_log.append(f"Traceback: {traceback.format_exc()}")
        error_md_final = gr.Markdown(f"❌ Train/Load Error: {str(e)}",elem_classes="status_box status_error")
        yield(error_md_final,None,"",None,None,None,None,gr.update(choices=[],value=None),gr.update(choices=[],value=None),None,gr.update(visible=False),gr.update(selected="train_tab"), "\n".join(training_log))

def update_confusion_matrix_plot(selected_model_for_cm):
    global y_test, all_trained_models, current_target_disease, X_test
    if y_test is None or not selected_model_for_cm or selected_model_for_cm not in all_trained_models or X_test is None: return gr.update(value=None, visible=False)
    model = all_trained_models[selected_model_for_cm]; y_pred = model.predict(X_test)
    inv_map = label_map_diabetes_inv if current_target_disease == "HadDiabetes" else {0: "No", 1: "Yes"}
    fig_cm = plot_confusion_matrix_custom(y_test, y_pred, selected_model_for_cm, current_target_disease, inv_map)
    return gr.update(value=fig_cm, visible=True)

def extract_rules_tab():
    global tree, X_train, rules, current_target_disease, model_feature_names
    if tree is None:
        return (gr.update(value=f"⚠️ Decision Tree model for `{current_target_disease}` is not available (not trained or not loaded correctly). Please train/load models first.",elem_classes="status_box status_warning"), gr.update(selected="train_tab"))
    try:
        if not model_feature_names and not hasattr(tree, 'feature_names_in_'):
             return (gr.update(value=f"⚠️ Feature names not available for Decision Tree rule extraction. This might happen if the model was loaded without its context or training data context is lost. Please re-run split and train steps.",elem_classes="status_box status_warning"), gr.update(selected="train_tab"))
        current_features_for_rules = None
        if hasattr(tree, 'feature_names_in_'):
            current_features_for_rules = tree.feature_names_in_
        elif model_feature_names:
            current_features_for_rules = model_feature_names
        else:
             return (gr.update(value=f"⚠️ Critical: Could not determine feature names for rule extraction.",elem_classes="status_box status_error"), gr.update(selected="train_tab"))
        rules = export_text(tree, feature_names=list(current_features_for_rules))
        rules_output_string = (f"Decision Rules for Target: {current_target_disease}\n(Model: Decision Tree)\n\n{rules[:3000]}...\n---END PREVIEW (Full rules can be long)---")
        return gr.update(value=rules_output_string), gr.update(selected="predict_tab")
    except Exception as e:
        rules = None
        return gr.update(value=f"❌ Error extracting rules: {str(e)}\nTrace: {traceback.format_exc()}",elem_classes="status_box status_error"), gr.update(selected="rules_tab")

def apply_preset_to_form_fields_revised(profile_text, *current_generic_textbox_vals):
    global form_creation_helper_data, SPECIFIC_INPUT_CONCEPTUAL_NAMES
    parsed_profile = parse_preset_profile_for_form(profile_text)
    updates = {key: gr.update() for key in SPECIFIC_INPUT_CONCEPTUAL_NAMES.keys()}
    if 'age' in parsed_profile: updates["age"] = gr.update(value=parsed_profile.get('age'))
    if 'bmi' in parsed_profile: updates["bmi"] = gr.update(value=parsed_profile.get('bmi'))
    sex_conceptual_name = SPECIFIC_INPUT_CONCEPTUAL_NAMES["sex"]
    if 'sex' in parsed_profile and sex_conceptual_name in form_creation_helper_data and form_creation_helper_data[sex_conceptual_name]['type'] == 'categorical':
        options = form_creation_helper_data[sex_conceptual_name]['options']
        parsed_val = parsed_profile['sex']
        matched_option = next((opt for opt in options if parsed_val.lower() in opt.lower()), None)
        if matched_option: updates["sex"] = gr.update(value=matched_option)
    smoked_conceptual_name = SPECIFIC_INPUT_CONCEPTUAL_NAMES["smoked_100_cigs"]
    if 'smoked_100_cigs' in parsed_profile and smoked_conceptual_name in form_creation_helper_data and form_creation_helper_data[smoked_conceptual_name]['type'] == 'categorical':
        options = form_creation_helper_data[smoked_conceptual_name]['options']
        parsed_val = parsed_profile['smoked_100_cigs']
        if parsed_val in options:
            updates["smoked_100_cigs"] = gr.update(value=parsed_val)
    num_generic_fields_in_ui = MAX_PREDICTION_FORM_FIELDS
    generic_updates_tuple = tuple(gr.update() for _ in range(num_generic_fields_in_ui))
    return (updates["state"], updates["age"], updates["sex"], updates["bmi"],
            updates["general_health"], updates["physical_activities"],
            updates["sleep_hours"], updates["smoked_100_cigs"], updates["ecig_usage"],
            updates["race_ethnicity"],
            updates["deaf"], updates["blind"], updates["difficulty_concentrating"],
            updates["difficulty_walking"], updates["difficulty_dressing"], updates["difficulty_errands"],
            updates["chest_scan"], updates["hiv_testing"], updates["flu_vax"],
            updates["pneumo_vax"], updates["high_risk_last_year"], updates["covid_pos"]
            ) + generic_updates_tuple

def setup_prediction_form_revised_fn():
    global model_feature_names, X_train, current_target_disease, form_creation_helper_data, SPECIFIC_INPUT_CONCEPTUAL_NAMES
    target_display_md = f"**Targeting for Prediction:** `{current_target_disease}` (Using features from last model training/loading for this target)"
    specific_updates = {key: gr.update(visible=False, choices=[], value=None) for key in SPECIFIC_INPUT_CONCEPTUAL_NAMES.keys()}
    specific_updates["age"] = gr.update(visible=False, value=None)
    specific_updates["bmi"] = gr.update(visible=False, value=None)
    specific_updates["sleep_hours"] = gr.update(visible=False, value=None)
    generic_field_updates = [gr.update(visible=False, value="", label="Feature") for _ in range(MAX_PREDICTION_FORM_FIELDS)]
    specific_inputs_group_upd = gr.update(visible=False)
    generic_form_fields_group_upd = gr.update(visible=False)
    error_html_upd = gr.HTML("<p class='status_box status_warning'>Model features not available. Run 'Split Data' (Tab 3) and 'Train & Evaluate' (Tab 4) first for the selected target.</p>", visible=True)
    if not model_feature_names:
        return (target_display_md,
                specific_updates["state"], specific_updates["age"], specific_updates["sex"], specific_updates["bmi"],
                specific_updates["general_health"], specific_updates["physical_activities"],
                specific_updates["sleep_hours"], specific_updates["smoked_100_cigs"], specific_updates["ecig_usage"],
                specific_updates["race_ethnicity"],
                specific_updates["deaf"], specific_updates["blind"], specific_updates["difficulty_concentrating"],
                specific_updates["difficulty_walking"], specific_updates["difficulty_dressing"], specific_updates["difficulty_errands"],
                specific_updates["chest_scan"], specific_updates["hiv_testing"], specific_updates["flu_vax"],
                specific_updates["pneumo_vax"], specific_updates["high_risk_last_year"], specific_updates["covid_pos"]
                ) + tuple(generic_field_updates) + \
               (specific_inputs_group_upd, generic_form_fields_group_upd, error_html_upd)
    error_html_upd = gr.HTML("", visible=False)
    specific_inputs_group_upd = gr.update(visible=True)
    def configure_specific_input(ui_key, label, default_val_index=0):
        concept_name_cleaned = SPECIFIC_INPUT_CONCEPTUAL_NAMES[ui_key]
        if concept_name_cleaned in form_creation_helper_data:
            item_info = form_creation_helper_data[concept_name_cleaned]
            if item_info["type"] == "categorical":
                options = item_info["options"]
                default_val = options[default_val_index] if options and len(options) > default_val_index else (options[0] if options else None)
                specific_updates[ui_key] = gr.update(label=label, choices=options, value=default_val, visible=True, interactive=True)
            elif item_info["type"] == "numerical":
                default_val_num = round(item_info.get('mean', 7 if ui_key=="sleep_hours" else (25 if ui_key=="bmi" else 0)))
                specific_updates[ui_key] = gr.update(label=label, value=default_val_num, visible=True, interactive=True)
        elif ui_key in ["bmi", "sleep_hours"]:
            default_val_num = 7 if ui_key=="sleep_hours" else 25
            specific_updates[ui_key] = gr.update(label=label, value=default_val_num, visible=True, interactive=True)
    configure_specific_input("state", "State")
    specific_updates["age"] = gr.update(label="Age (Years)", value=45, visible=True, interactive=True)
    configure_specific_input("sex", "Sex")
    configure_specific_input("bmi", "BMI")
    configure_specific_input("general_health", "General Health")
    pa_concept = SPECIFIC_INPUT_CONCEPTUAL_NAMES["physical_activities"]
    if pa_concept in form_creation_helper_data and form_creation_helper_data[pa_concept]['type'] == 'categorical':
        pa_opts = form_creation_helper_data[pa_concept]['options']
        specific_updates["physical_activities"] = gr.update(label="Exercise Last 30 Days", choices=pa_opts, value=pa_opts[0] if pa_opts else None, visible=True, interactive=True)
    else:
        specific_updates["physical_activities"] = gr.update(label="Exercise Last 30 Days", choices=["No", "Yes"], value="No", visible=True, interactive=True)
    configure_specific_input("sleep_hours", "Avg. Sleep Hours")
    s100_concept = SPECIFIC_INPUT_CONCEPTUAL_NAMES["smoked_100_cigs"]
    if s100_concept in form_creation_helper_data and form_creation_helper_data[s100_concept]['type'] == 'categorical':
        s100_opts = form_creation_helper_data[s100_concept]['options']
        specific_updates["smoked_100_cigs"] = gr.update(label="Smoked >100 Cigs", choices=s100_opts, value=s100_opts[0] if s100_opts else None, visible=True, interactive=True)
    else:
        specific_updates["smoked_100_cigs"] = gr.update(label="Smoked >100 Cigs", choices=["No", "Yes"], value="No", visible=True, interactive=True)
    configure_specific_input("ecig_usage", "E-Cigarette Usage")
    configure_specific_input("race_ethnicity", "Race/Ethnicity")
    yes_no_options_default = ["No", "Yes"]
    for key, label_text in [
        ("deaf", "Deaf/Hard of Hearing"), ("blind", "Blind/Vision Difficulty"),
        ("difficulty_concentrating", "Difficulty Concentrating"), ("difficulty_walking", "Difficulty Walking"),
        ("difficulty_dressing", "Difficulty Dressing/Bathing"), ("difficulty_errands", "Difficulty Errands"),
        ("chest_scan", "Had Chest Scan"), ("hiv_testing", "HIV Test Ever"),
        ("flu_vax", "Flu Vaccine Last 12 Mo"), ("pneumo_vax", "Pneumonia Vaccine Ever"),
        ("high_risk_last_year", "High Risk Past Year"), ("covid_pos", "Tested Positive for COVID-19")
    ]:
        concept_name = SPECIFIC_INPUT_CONCEPTUAL_NAMES[key]
        options_to_use = yes_no_options_default
        default_value = yes_no_options_default[0]
        if concept_name in form_creation_helper_data and form_creation_helper_data[concept_name]['type'] == 'categorical':
            helper_opts = form_creation_helper_data[concept_name]['options']
            if helper_opts and all(isinstance(o, str) for o in helper_opts) and len(helper_opts) >= 2:
                no_like_val = next((o for o in helper_opts if "no" in o.lower()), None)
                if no_like_val: default_value = no_like_val
                else: default_value = helper_opts[0]
                options_to_use = helper_opts
        specific_updates[key] = gr.update(label=label_text, choices=options_to_use, value=default_value, visible=True, interactive=True)
    current_generic_idx = 0
    for mf_name in model_feature_names:
        is_handled_by_specific_ui = False
        if mf_name == SPECIFIC_INPUT_CONCEPTUAL_NAMES.get("bmi") or \
           mf_name == SPECIFIC_INPUT_CONCEPTUAL_NAMES.get("sleep_hours"):
            is_handled_by_specific_ui = True
        elif mf_name.lower().startswith("agecategory"):
            is_handled_by_specific_ui = True
        else:
            for concept_key, concept_val_cleaned in SPECIFIC_INPUT_CONCEPTUAL_NAMES.items():
                if concept_key in ["age", "bmi", "sleep_hours"]: continue
                if mf_name.startswith(concept_val_cleaned) and \
                   mf_name != concept_val_cleaned and \
                   concept_val_cleaned in form_creation_helper_data and \
                   form_creation_helper_data[concept_val_cleaned]["type"] == "categorical":
                    is_handled_by_specific_ui = True
                    break
                if concept_val_cleaned in form_creation_helper_data and \
                   form_creation_helper_data[concept_val_cleaned]["type"] == "categorical" and \
                   (mf_name == concept_val_cleaned + "Yes" or mf_name == concept_val_cleaned + "No"):
                    is_handled_by_specific_ui = True
                    break
        if not is_handled_by_specific_ui:
            if current_generic_idx < MAX_PREDICTION_FORM_FIELDS:
                label_text = mf_name.replace("_", " ").title()
                default_value_str = "0"
                placeholder_text = "Enter value (e.g., 0 or 1 if binary)"
                if X_train is not None and mf_name in X_train.columns:
                    col_data = X_train[mf_name]
                    if not (col_data.nunique(dropna=False) <= 2 and set(col_data.dropna().unique()) <= {0, 1, 0.0, 1.0}):
                        if pd.api.types.is_numeric_dtype(col_data.dtype) and not col_data.empty:
                            try:
                                default_value_str = str(round(col_data.mean(), 2))
                                placeholder_text = f"e.g., {default_value_str} (numeric)"
                            except TypeError:
                                pass
                generic_field_updates[current_generic_idx] = gr.update(label=label_text, value=default_value_str, placeholder=placeholder_text, visible=True, interactive=True)
                current_generic_idx += 1
    for i in range(current_generic_idx, MAX_PREDICTION_FORM_FIELDS):
        generic_field_updates[i] = gr.update(visible=False, value="", label="Feature")
    generic_form_fields_group_upd = gr.update(visible=current_generic_idx > 0)
    return (target_display_md,
            specific_updates["state"], specific_updates["age"], specific_updates["sex"], specific_updates["bmi"],
            specific_updates["general_health"], specific_updates["physical_activities"],
            specific_updates["sleep_hours"], specific_updates["smoked_100_cigs"], specific_updates["ecig_usage"],
            specific_updates["race_ethnicity"],
            specific_updates["deaf"], specific_updates["blind"], specific_updates["difficulty_concentrating"],
            specific_updates["difficulty_walking"], specific_updates["difficulty_dressing"], specific_updates["difficulty_errands"],
            specific_updates["chest_scan"], specific_updates["hiv_testing"], specific_updates["flu_vax"],
            specific_updates["pneumo_vax"], specific_updates["high_risk_last_year"], specific_updates["covid_pos"]
            ) + tuple(generic_field_updates) + \
           (specific_inputs_group_upd, generic_form_fields_group_upd, error_html_upd)

def run_prediction_tab_form_revised(selected_model_for_pred,
                             state_ui, age_ui, sex_ui, bmi_ui,
                             general_health_ui, physical_activities_ui,
                             sleep_hours_ui, smoked_100_cigs_ui, ecig_usage_ui,
                             race_ethnicity_ui,
                             deaf_ui, blind_ui, difficulty_concentrating_ui,
                             difficulty_walking_ui, difficulty_dressing_ui, difficulty_errands_ui,
                             chest_scan_ui, hiv_testing_ui, flu_vax_ui,
                             pneumo_vax_ui, high_risk_last_year_ui, covid_pos_ui,
                             *generic_feature_values_tuple):
    global all_trained_models, current_target_disease, model_feature_names, X_train, form_creation_helper_data, SPECIFIC_INPUT_CONCEPTUAL_NAMES
    if not selected_model_for_pred or selected_model_for_pred not in all_trained_models:
        return gr.Markdown("⚠️ Model not selected or not trained/loaded properly for the current target. Please go to Tab 4.", elem_classes="status_box status_warning")
    if not model_feature_names:
        return gr.Markdown("⚠️ Model feature list (expected inputs) is not available. Ensure model training (Tab 4) was successful for the current target.", elem_classes="status_box status_warning")
    model = all_trained_models[selected_model_for_pred]
    input_data = pd.Series(0.0, index=model_feature_names)
    def process_categorical_to_one_hot(ui_value, conceptual_name_key_in_spec, input_data_series_ref):
        if ui_value is None or ui_value == "": return
        conceptual_cleaned_name = SPECIFIC_INPUT_CONCEPTUAL_NAMES[conceptual_name_key_in_spec]
        cleaned_val_part = str(ui_value).replace(" ", "").replace("/", "").replace("-", "").replace("(", "").replace(")", "")
        potential_feature_name = conceptual_cleaned_name + cleaned_val_part
        if potential_feature_name in input_data_series_ref.index:
            input_data_series_ref[potential_feature_name] = 1.0
    process_categorical_to_one_hot(state_ui, "state", input_data)
    if age_ui is not None:
        try:
            age = int(age_ui)
            age_cat_feature_name = get_age_category_feature(age, model_feature_names)
            if age_cat_feature_name and age_cat_feature_name in input_data.index:
                input_data[age_cat_feature_name] = 1.0
        except ValueError: return gr.Markdown("❌ Invalid Age. Please enter a number.", elem_classes="status_box status_error")
    process_categorical_to_one_hot(sex_ui, "sex", input_data)
    bmi_concept_name = SPECIFIC_INPUT_CONCEPTUAL_NAMES["bmi"]
    if bmi_ui is not None:
        if bmi_concept_name in input_data.index:
            try: input_data[bmi_concept_name] = float(bmi_ui)
            except ValueError: return gr.Markdown("❌ Invalid BMI. Please enter a number.", elem_classes="status_box status_error")
    process_categorical_to_one_hot(general_health_ui, "general_health", input_data)
    process_categorical_to_one_hot(physical_activities_ui, "physical_activities", input_data)
    sh_concept_name = SPECIFIC_INPUT_CONCEPTUAL_NAMES["sleep_hours"]
    if sleep_hours_ui is not None:
        if sh_concept_name in input_data.index:
            try: input_data[sh_concept_name] = float(sleep_hours_ui)
            except ValueError: return gr.Markdown("❌ Invalid Sleep Hours. Please enter a number.", elem_classes="status_box status_error")
    process_categorical_to_one_hot(smoked_100_cigs_ui, "smoked_100_cigs", input_data)
    process_categorical_to_one_hot(ecig_usage_ui, "ecig_usage", input_data)
    process_categorical_to_one_hot(race_ethnicity_ui, "race_ethnicity", input_data)
    yes_no_radios_map = {
        "deaf": deaf_ui, "blind": blind_ui, "difficulty_concentrating": difficulty_concentrating_ui,
        "difficulty_walking": difficulty_walking_ui, "difficulty_dressing": difficulty_dressing_ui,
        "difficulty_errands": difficulty_errands_ui, "chest_scan": chest_scan_ui,
        "hiv_testing": hiv_testing_ui, "flu_vax": flu_vax_ui, "pneumo_vax": pneumo_vax_ui,
        "high_risk_last_year": high_risk_last_year_ui, "covid_pos": covid_pos_ui
    }
    for key_in_spec, ui_val in yes_no_radios_map.items():
        process_categorical_to_one_hot(ui_val, key_in_spec, input_data)
    generic_features_in_form_order = []
    temp_model_features_copy = model_feature_names.copy()
    handled_specifically = set()
    if SPECIFIC_INPUT_CONCEPTUAL_NAMES.get("bmi") in temp_model_features_copy: handled_specifically.add(SPECIFIC_INPUT_CONCEPTUAL_NAMES.get("bmi"))
    if SPECIFIC_INPUT_CONCEPTUAL_NAMES.get("sleep_hours") in temp_model_features_copy: handled_specifically.add(SPECIFIC_INPUT_CONCEPTUAL_NAMES.get("sleep_hours"))
    for mf_name in temp_model_features_copy:
        if mf_name.lower().startswith("agecategory"): handled_specifically.add(mf_name)
        else:
            for concept_key, concept_val_cleaned in SPECIFIC_INPUT_CONCEPTUAL_NAMES.items():
                if concept_key in ["age", "bmi", "sleep_hours"]: continue
                if mf_name.startswith(concept_val_cleaned) and mf_name != concept_val_cleaned and \
                   concept_val_cleaned in form_creation_helper_data and \
                   form_creation_helper_data[concept_val_cleaned]["type"] == "categorical":
                    handled_specifically.add(mf_name)
                    break
                if concept_val_cleaned in form_creation_helper_data and \
                   form_creation_helper_data[concept_val_cleaned]["type"] == "categorical" and \
                   (mf_name == concept_val_cleaned + "Yes" or mf_name == concept_val_cleaned + "No"):
                     handled_specifically.add(mf_name)
                     break
    generic_features_in_form_order = [mf for mf in model_feature_names if mf not in handled_specifically]
    for idx, generic_mf_name in enumerate(generic_features_in_form_order):
        if idx < len(generic_feature_values_tuple):
            val_str = str(generic_feature_values_tuple[idx]).strip()
            if not val_str:
                is_binary_like = False
                if X_train is not None and generic_mf_name in X_train.columns:
                    col_unique_vals = X_train[generic_mf_name].dropna().unique()
                    if len(col_unique_vals) <= 2 and set(col_unique_vals) <= {0, 1, 0.0, 1.0}:
                        is_binary_like = True
                if is_binary_like:
                    input_data[generic_mf_name] = 0.0
                else:
                    try:
                        input_data[generic_mf_name] = float(val_str)
                    except ValueError:
                        return gr.Markdown(f"❌ Invalid numeric value for generic feature `{generic_mf_name}`: '{val_str}'. Please enter a number.", elem_classes="status_box status_error")
    try:
        # Ensure order of columns in input_df matches model's expectation (model_feature_names)
        input_df = pd.DataFrame([input_data[model_feature_names].values], columns=model_feature_names)
        # Make prediction
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]
        
        inv_map = label_map_diabetes_inv if current_target_disease == "HadDiabetes" else {0: "No", 1: "Yes"}
        predicted_label = inv_map.get(prediction, "Unknown")
        
        # Format output
        conf_score = prediction_proba[prediction] * 100
        result_html = f"""
        <div style='padding: 15px; border-radius: 8px; border: 1px solid #ddd; background-color: #f9f9f9;'>
            <h4 style='color: #333; border-bottom: 2px solid #007bff; padding-bottom: 5px;'>Prediction Result</h4>
            <p><strong>Selected Model:</strong> {selected_model_for_pred}</p>
            <p><strong>Predicted Outcome for <span style='color:#2c3e50; font-weight:bold;'>{current_target_disease}</span>:</strong> <span class='prediction-outcome'>{predicted_label}</span></p>
            <p><strong>Confidence Score:</strong> {conf_score:.2f}%</p>
        </div>
        """
        return gr.update(value=result_html)

    except Exception as e:
        tb_str = traceback.format_exc()
        return gr.update(value=f"<p style='color:red;font-weight:bold;'>Prediction Error:</p><pre>{str(e)}\n{tb_str}</pre>")


def update_profile_textbox_from_preset(preset_text):
    return gr.update(value=preset_text)

def generate_advice_for_models_tab(profile, selected_llms, progress=gr.Progress(track_tqdm=True)):
    global generated_advice_text, advice_evaluation_text, rules, current_target_disease, MODEL_CHOICES_MAP
    if not profile or not selected_llms:
        return gr.update(value="Please provide a patient profile and select at least one LLM."), gr.update(selected="advise_eval_tab"), "", "", ""

    advice_evaluation_text = ""
    advice_outputs = []
    
    # Construct a more detailed prompt
    rules_context = f"For context, here are some decision rules related to predicting '{current_target_disease}':\n{rules[:1500]}..." if rules else "No decision rules are currently available for context."
    prompt = f"Based on the following patient profile and decision-rule context, provide clear, empathetic, and actionable health advice. Do NOT give a diagnosis. Emphasize consulting a doctor.\n\n**Patient Profile:**\n{profile}\n\n**Contextual Rules:**\n{rules_context}"

    for model_name in progress.tqdm(selected_llms, desc="Generating Advice from LLMs"):
        advice_outputs.append(f"### 🤖 Advice from {model_name}\n---\n")
        
        provider, model_id = MODEL_CHOICES_MAP.get(model_name, (None, None))
        if not provider:
            response = f"Error: Could not find '{model_name}' in the configuration."
        else:
            response = get_response(provider, model_id, prompt)

        advice_outputs.append(response + "\n\n")

    full_advice = "".join(advice_outputs)
    generated_advice_text = full_advice
    content_for_saving = generated_advice_text
    
    return gr.update(value=full_advice), gr.update(selected="advise_eval_tab"), generated_advice_text, "", content_for_saving

def evaluate_current_advice_with_llm(advice_to_eval, llm_for_eval):
    global generated_advice_text, advice_evaluation_text, MODEL_CHOICES_MAP
    if not advice_to_eval:
        return "There is no advice to evaluate. Please generate advice first.", "", "No content to save."
    if not llm_for_eval:
        return "Please select an LLM to perform the evaluation.", "", "No content to save."

    prompt = f"Please critically evaluate the following health advice for clarity, safety, and actionability. Provide a summary of its strengths and weaknesses. The advice is:\n\n---\n\n{advice_to_eval}"
    
    provider, model_id = MODEL_CHOICES_MAP.get(llm_for_eval, (None, None))
    if not provider:
        evaluation_response = f"Error: Could not find evaluator model '{llm_for_eval}' in configuration."
    else:
        evaluation_response = get_response(provider, model_id, prompt)
    
    advice_evaluation_text = f"### ⚖️ Evaluation from {llm_for_eval}\n---\n{evaluation_response}"
    content_for_saving = generated_advice_text + "\n\n" + advice_evaluation_text
    
    return advice_evaluation_text, evaluation_response, content_for_saving

def save_advice_and_evaluation_tab(formats, content):
    filepath, message = save_report_files(formats, content)
    status_md = gr.Markdown(message, elem_classes="status_box status_success" if filepath else "status_box status_error")
    return gr.update(value=filepath, visible=bool(filepath)), status_md 