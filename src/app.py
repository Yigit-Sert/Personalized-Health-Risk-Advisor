import gradio as gr
import warnings

from .callbacks import (
    load_dataset_tab, clean_encode_tab, split_data_tab, update_target_viz,
    train_model_tab, update_confusion_matrix_plot, extract_rules_tab,
    setup_prediction_form_revised_fn, apply_preset_to_form_fields_revised, run_prediction_tab_form_revised,
    update_profile_textbox_from_preset, generate_advice_for_models_tab,
    evaluate_current_advice_with_llm, save_advice_and_evaluation_tab,
    current_target_disease
)
from .config import *

warnings.filterwarnings("ignore")

def create_gradio_app():
    with gr.Blocks(css=APP_CSS, theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🏥 Personalized Health Insights, Prediction & LLM Advisor", elem_id="app_title")
        gr.Markdown("Follow tabs sequentially. API keys are needed for LLM features (Tab 7). Models will be saved to/loaded from a 'saved_models' directory.")

        # --- Gradio State Management ---
        y_train_counts_state = gr.State()
        y_test_counts_state = gr.State()
        inv_map_state = gr.State()
        current_target_disease_state = gr.State(current_target_disease)
        generated_advice_text_state = gr.State("")
        generated_evaluation_text_state = gr.State("")
        content_for_saving_state = gr.State("")

        # --- UI Layout ---
        main_tabs = gr.Tabs(elem_id="main_app_tabs")
        with main_tabs:
            with gr.Tab("1. Load Dataset", id="load_tab"):
                gr.Markdown("### Step 1: Load Health Indicators Dataset"); load_btn = gr.Button("🚀 Load Dataset", variant="primary")
                load_status_md = gr.Markdown()
                with gr.Accordion("Explore Loaded Data", open=False) as load_details_accordion:
                    with gr.Group(elem_classes="info-box"): gr.Markdown("#### Quick Peek: First 5 Rows",elem_classes="info-box-header"); df_preview = gr.DataFrame(interactive=False, wrap=True)
                    with gr.Group(elem_classes="info-box"): gr.Markdown("#### Dataset Shape",elem_classes="info-box-header"); df_shape_md = gr.Markdown()
                    with gr.Group(elem_classes="info-box"): gr.Markdown("#### Column Information",elem_classes="info-box-header"); df_info_summary_md = gr.Markdown(); df_info_table = gr.DataFrame(interactive=False, wrap=True)

            with gr.Tab("2. Clean & Prepare", id="clean_tab"):
                gr.Markdown("### Step 2: Clean, Encode, and Prepare Data"); enc_btn = gr.Button("✨ Clean, Map & Encode", variant="primary")
                encode_status_md = gr.Markdown()
                with gr.Accordion("Explore Processed Data", open=False) as encode_details_accordion:
                    with gr.Group(elem_classes="info-box"): gr.Markdown("#### Quick Peek: Processed",elem_classes="info-box-header"); df_enc_preview = gr.DataFrame(interactive=False, wrap=True)
                    with gr.Group(elem_classes="info-box"): gr.Markdown("#### Processed Shape",elem_classes="info-box-header"); df_enc_shape_md = gr.Markdown()
                    with gr.Group(elem_classes="info-box"): gr.Markdown("#### Processed Column Info",elem_classes="info-box-header"); df_enc_info_summary_md = gr.Markdown(); df_enc_info_table = gr.DataFrame(interactive=False, wrap=True)
                    with gr.Group(elem_classes="info-box"): gr.Markdown("#### Summary of Changes",elem_classes="info-box-header"); df_enc_changes_md = gr.Markdown()

            with gr.Tab("3. Split Data", id="split_tab"):
                gr.Markdown("### Step 3: Select Target & Split Data")
                target_disease_dd_split = gr.Dropdown(label="🎯 Select Target Disease", choices=[], interactive=True, info="This is the condition the models will predict. Ensure data is cleaned (Tab 2) first.")
                split_status_md = gr.Markdown()
                with gr.Row(): y_train_dist_type_radio = gr.Radio(choices=["Bar Chart","Pie Chart","Table"],value="Bar Chart",label="Train Target Viz",interactive=True); y_test_dist_type_radio = gr.Radio(choices=["Bar Chart","Pie Chart","Table"],value="Bar Chart",label="Test Target Viz",interactive=True)
                split_btn = gr.Button("✂️ Split Data", variant="primary")
                with gr.Accordion("Explore Split Data & Target Distributions", open=False) as split_details_accordion:
                    with gr.Group(elem_classes="info-box"): split_target_variable_display = gr.Markdown(); split_shapes_md = gr.Markdown()
                    gr.Markdown("#### Training Set Target Distribution")
                    with gr.Row():
                        train_dist_plot = gr.Plot(scale=2)
                        train_dist_table_display = gr.DataFrame(wrap=True,scale=1,visible=False)
                    gr.Markdown("#### Testing Set Target Distribution")
                    with gr.Row():
                        test_dist_plot = gr.Plot(scale=2)
                        test_dist_table_display = gr.DataFrame(wrap=True,scale=1,visible=False)

            with gr.Tab("4. Train & Evaluate", id="train_tab"):
                gr.Markdown("### Step 4: Train (or Load Pre-trained) Predictive Models & Evaluate")
                gr.Markdown("Models will be loaded from disk if previously trained for the current target and data structure. Otherwise, they will be trained and saved. Evaluation metrics shown are for the active models.")
                train_btn = gr.Button("🧠 Train/Load & Evaluate Models", variant="primary")
                train_status_md = gr.Markdown()
                training_log_md = gr.Markdown(elem_id="training_log_md_id", value="Training log will appear here...")
                with gr.Accordion("Model Performance & Insights", open=True) as train_details_accordion:
                    gr.Markdown("#### Comparative Model Performance Metrics"); results_df_display = gr.DataFrame(wrap=True, label="Model Metrics Table")
                    with gr.Group(elem_classes="info-box"): gr.Markdown("#### Best Model Summary (based on current evaluation)",elem_classes="info-box-header"); best_models_summary_md = gr.Markdown()
                    gr.Markdown("#### Overall Model Comparison Plot"); model_comparison_plot = gr.Plot(label="Metrics Comparison Plot")
                    gr.Markdown("---"); gr.Markdown("#### Feature Importances (for tree-based models)")
                    with gr.Tabs():
                        with gr.Tab("Decision Tree"): feat_imp_dt_plot=gr.Plot(label="DT Feature Importances")
                        with gr.Tab("Random Forest"): feat_imp_rf_plot=gr.Plot(label="RF Feature Importances")
                        with gr.Tab("Gradient Boosting"): feat_imp_gb_plot=gr.Plot(label="GB Feature Importances")
                    gr.Markdown("---"); gr.Markdown("#### Confusion Matrix"); cm_model_select_dd = gr.Dropdown(label="Select Model for Confusion Matrix", choices=[], interactive=True); confusion_matrix_plot = gr.Plot(label="Confusion Matrix Plot")

            with gr.Tab("5. Decision Rules", id="rules_tab"):
                gr.Markdown("### Step 5: Extract Rules from Decision Tree"); rules_btn = gr.Button("📜 Extract Decision Rules", variant="primary")
                rules_disp_tb = gr.Textbox(label="Extracted Decision Rules (Preview from active Decision Tree model)", lines=20, interactive=False, show_copy_button=True)

            with gr.Tab("6. Make a Prediction", id="predict_tab"):
                gr.Markdown("### Step 6: Get a Prediction based on an Active Model"); current_target_display_pred_md = gr.Markdown()
                with gr.Row():
                    prediction_model_select = gr.Dropdown(label="Select Active Model for Prediction", choices=[], interactive=True, scale=1, info="Models available from Tab 4 for the current target.")
                    pred_preset_profile_dd = gr.Dropdown(label="Use Preset Profile (Optional)", choices=PRESET_PROFILES, interactive=True, scale=2)

                generate_form_btn = gr.Button("📝 Generate/Reset Input Form for Current Target's Features")
                prediction_form_error_html = gr.HTML(visible=False)
                with gr.Group(visible=False) as prediction_specific_inputs_group:
                    gr.Markdown("#### Patient Details & Health Indicators (Specific Inputs):")
                    with gr.Row(): state_input_pred = gr.Dropdown(label="State", visible=False, interactive=True); age_input_pred = gr.Number(label="Age (Years)", visible=False, interactive=True); sex_input_pred = gr.Dropdown(label="Sex", visible=False, interactive=True)
                    with gr.Row(): bmi_input_pred = gr.Number(label="BMI", visible=False, interactive=True); general_health_input_pred = gr.Dropdown(label="General Health", visible=False, interactive=True)
                    with gr.Row(): physical_activities_input_pred = gr.Radio(label="Exercise Last 30 Days", visible=False, interactive=True); sleep_hours_input_pred = gr.Number(label="Avg. Sleep Hours", visible=False, interactive=True); smoked_100_cigs_input_pred = gr.Radio(label="Smoked >100 Cigs", visible=False, interactive=True)
                    with gr.Row(): ecig_usage_input_pred = gr.Dropdown(label="E-Cigarette Usage", visible=False, interactive=True); race_ethnicity_input_pred = gr.Dropdown(label="Race/Ethnicity", visible=False, interactive=True)
                    gr.Markdown("##### Other Health Conditions & Behaviors (Yes/No):")
                    with gr.Row(): deaf_input_pred = gr.Radio(label="Deaf/Hard of Hearing", visible=False, interactive=True); blind_input_pred = gr.Radio(label="Blind/Vision Difficulty", visible=False, interactive=True); difficulty_concentrating_input_pred = gr.Radio(label="Difficulty Concentrating", visible=False, interactive=True)
                    with gr.Row(): difficulty_walking_input_pred = gr.Radio(label="Difficulty Walking", visible=False, interactive=True); difficulty_dressing_input_pred = gr.Radio(label="Difficulty Dressing/Bathing", visible=False, interactive=True); difficulty_errands_input_pred = gr.Radio(label="Difficulty Errands", visible=False, interactive=True)
                    with gr.Row(): chest_scan_input_pred = gr.Radio(label="Had Chest Scan", visible=False, interactive=True); hiv_testing_input_pred = gr.Radio(label="HIV Test Ever", visible=False, interactive=True); flu_vax_input_pred = gr.Radio(label="Flu Vaccine Last 12 Mo", visible=False, interactive=True)
                    with gr.Row(): pneumo_vax_input_pred = gr.Radio(label="Pneumonia Vaccine Ever", visible=False, interactive=True); high_risk_last_year_input_pred = gr.Radio(label="High Risk Activity Past Year", visible=False, interactive=True); covid_pos_input_pred = gr.Radio(label="Tested Positive for COVID-19", visible=False, interactive=True)
                with gr.Group(visible=False) as prediction_form_fields_group:
                    gr.Markdown("##### Other Model Features (auto-populated with defaults or fill if needed):")
                    prediction_form_fields_inputs = [gr.Textbox(label=f"Generic Feature {i+1}", visible=False, interactive=True, elem_classes="prediction_input_field") for i in range(MAX_PREDICTION_FORM_FIELDS)]
                predict_btn = gr.Button("💡 Get Prediction", variant="primary"); prediction_output_md = gr.Markdown(label="Prediction Result")

            with gr.Tab("7. LLM Advice & Evaluation", id="advise_eval_tab"):
                gr.Markdown("### Step 7: Generate & Evaluate Health Advice using LLMs"); current_advice_target_md = gr.Markdown()
                gr.Markdown(
                    "💡 **Note:** Models labeled `(Local)` run on your device using Ollama and will consume CPU/RAM resources. "
                    "Other models use cloud-based API services.",
                    elem_classes="info-box"
                )
                with gr.Row():
                    with gr.Column(scale=2):
                        profile_in_tb = gr.Textbox(label="Enter Patient Profile for Advice", lines=5, placeholder="Example: 50-year-old male, smoker, BMI 30..."); profile_dd_advice = gr.Dropdown(choices=PRESET_PROFILES, label="Or, Use a Preset Profile")
                        model_cbg_advice = gr.CheckboxGroup(choices=list(MODEL_CHOICES_MAP.keys()), label="Select LLM(s) for Generating Advice", value=[list(MODEL_CHOICES_MAP.keys())[0]] if MODEL_CHOICES_MAP else [])
                        gen_adv_btn = gr.Button("💬 Generate Advice", variant="primary")
                    with gr.Column(scale=3): adv_disp_md = gr.Markdown(label="Generated Advice from LLMs (Raw Output)")
                gr.Markdown("---"); gr.Markdown("#### Evaluate Generated Advice (using another LLM)");
                with gr.Row():
                    llm_for_eval_dd_advise = gr.Dropdown(choices=list(MODEL_CHOICES_MAP.keys()), label="Select LLM for Evaluation", value=list(MODEL_CHOICES_MAP.keys())[1] if len(MODEL_CHOICES_MAP.keys()) > 1 else (list(MODEL_CHOICES_MAP.keys())[0] if MODEL_CHOICES_MAP else None), scale=1, info="Choose an LLM to critique the advice generated above.")
                    eval_advice_btn = gr.Button("🔎 Evaluate This Advice", scale=1)
                advice_evaluation_output_md = gr.Markdown(label="LLM's Evaluation of the Advice")

            with gr.Tab("8. Save Results", id="save_tab"):
                gr.Markdown("### Step 8: Save Generated Advice & Evaluation Report");
                gr.Markdown("The content below (combined advice from Tab 7 and its evaluation, if performed) will be saved in your chosen formats.");
                content_to_save_display_md = gr.Markdown(label="Content Preview for Saving (Advice + Evaluation)")
                save_format_cbg = gr.CheckboxGroup(choices=["Markdown (.md)", "Plain Text (.txt)", "HTML (.html)", "PDF (.pdf)"], label="Select File Format(s) to Save", value=["Markdown (.md)"])
                save_all_btn = gr.Button("💾 Save Report", variant="primary"); save_final_status_md = gr.Markdown()
                dl_final_file = gr.File(label="Download Your Report", visible=False, interactive=False)

        # === UI ELEMENT CONNECTIONS ===
        load_outputs = [load_status_md, load_details_accordion, df_preview, df_shape_md, df_info_table, df_info_summary_md, main_tabs, target_disease_dd_split]
        load_btn.click(fn=load_dataset_tab, inputs=None, outputs=load_outputs)

        encode_outputs = [encode_status_md, encode_details_accordion, df_enc_preview, df_enc_shape_md, df_enc_info_table, df_enc_info_summary_md, df_enc_changes_md,
                          target_disease_dd_split, main_tabs]
        enc_btn.click(fn=clean_encode_tab, inputs=None, outputs=encode_outputs)

        def update_advice_tab_target_text_on_dd_change(disease_name_from_split_tab_dd):
            current_target_disease_state.value = disease_name_from_split_tab_dd
            return f"**Advice & Evaluation will relate to: `{disease_name_from_split_tab_dd}`** (if rules are extracted for this target)." if disease_name_from_split_tab_dd else "**Advice Target: `No Target Selected in Tab 3`**."

        target_disease_dd_split.change(
            fn=update_advice_tab_target_text_on_dd_change,
            inputs=target_disease_dd_split,
            outputs=[current_advice_target_md]
        ).then(
            lambda disease: disease,
            inputs=target_disease_dd_split,
            outputs=current_target_disease_state
        )

        prediction_form_setup_outputs_list = [current_target_display_pred_md,
                                         state_input_pred, age_input_pred, sex_input_pred, bmi_input_pred,
                                         general_health_input_pred, physical_activities_input_pred,
                                         sleep_hours_input_pred, smoked_100_cigs_input_pred, ecig_usage_input_pred,
                                         race_ethnicity_input_pred,
                                         deaf_input_pred, blind_input_pred, difficulty_concentrating_input_pred,
                                         difficulty_walking_input_pred, difficulty_dressing_input_pred, difficulty_errands_input_pred,
                                         chest_scan_input_pred, hiv_testing_input_pred, flu_vax_input_pred,
                                         pneumo_vax_input_pred, high_risk_last_year_input_pred, covid_pos_input_pred
                                         ] + prediction_form_fields_inputs + \
                                        [prediction_specific_inputs_group, prediction_form_fields_group, prediction_form_error_html]

        split_outputs = [split_status_md, split_shapes_md, split_target_variable_display, train_dist_plot, train_dist_table_display, test_dist_plot, test_dist_table_display, y_train_counts_state, y_test_counts_state, inv_map_state, split_details_accordion, main_tabs]
        split_btn.click(fn=split_data_tab, inputs=[target_disease_dd_split, y_train_dist_type_radio, y_test_dist_type_radio], outputs=split_outputs).then(
            fn=setup_prediction_form_revised_fn, inputs=None, outputs=prediction_form_setup_outputs_list
        )
        y_train_dist_type_radio.change(fn=update_target_viz,inputs=[y_train_counts_state,current_target_disease_state,inv_map_state,y_train_dist_type_radio,gr.State("Train Set Distribution for")],outputs=[train_dist_plot,train_dist_table_display])
        y_test_dist_type_radio.change(fn=update_target_viz,inputs=[y_test_counts_state,current_target_disease_state,inv_map_state,y_test_dist_type_radio,gr.State("Test Set Distribution for")],outputs=[test_dist_plot,test_dist_table_display])

        train_outputs = [train_status_md, results_df_display, best_models_summary_md, model_comparison_plot, feat_imp_dt_plot, feat_imp_rf_plot, feat_imp_gb_plot, prediction_model_select, cm_model_select_dd, confusion_matrix_plot, train_details_accordion, main_tabs, training_log_md]
        train_btn.click(fn=train_model_tab, inputs=None, outputs=train_outputs).then(
            fn=setup_prediction_form_revised_fn, inputs=None, outputs=prediction_form_setup_outputs_list
        )
        cm_model_select_dd.change(fn=update_confusion_matrix_plot, inputs=[cm_model_select_dd], outputs=[confusion_matrix_plot])

        rules_btn.click(fn=extract_rules_tab, inputs=None, outputs=[rules_disp_tb, main_tabs])

        generate_form_btn.click(fn=setup_prediction_form_revised_fn, inputs=None, outputs=prediction_form_setup_outputs_list)

        apply_preset_input_components = [pred_preset_profile_dd] + prediction_form_fields_inputs
        apply_preset_output_components = [state_input_pred, age_input_pred, sex_input_pred, bmi_input_pred,
                                         general_health_input_pred, physical_activities_input_pred,
                                         sleep_hours_input_pred, smoked_100_cigs_input_pred, ecig_usage_input_pred,
                                         race_ethnicity_input_pred,
                                         deaf_input_pred, blind_input_pred, difficulty_concentrating_input_pred,
                                         difficulty_walking_input_pred, difficulty_dressing_input_pred, difficulty_errands_input_pred,
                                         chest_scan_input_pred, hiv_testing_input_pred, flu_vax_input_pred,
                                         pneumo_vax_input_pred, high_risk_last_year_input_pred, covid_pos_input_pred
                                         ] + prediction_form_fields_inputs

        pred_preset_profile_dd.change(fn=apply_preset_to_form_fields_revised,
                                      inputs=apply_preset_input_components,
                                      outputs=apply_preset_output_components)

        predict_btn_input_components = [prediction_model_select,
                                       state_input_pred, age_input_pred, sex_input_pred, bmi_input_pred,
                                       general_health_input_pred, physical_activities_input_pred,
                                       sleep_hours_input_pred, smoked_100_cigs_input_pred, ecig_usage_input_pred,
                                       race_ethnicity_input_pred,
                                       deaf_input_pred, blind_input_pred, difficulty_concentrating_input_pred,
                                       difficulty_walking_input_pred, difficulty_dressing_input_pred, difficulty_errands_input_pred,
                                       chest_scan_input_pred, hiv_testing_input_pred, flu_vax_input_pred,
                                       pneumo_vax_input_pred, high_risk_last_year_input_pred, covid_pos_input_pred
                                       ] + prediction_form_fields_inputs
        predict_btn.click(fn=run_prediction_tab_form_revised, inputs=predict_btn_input_components, outputs=[prediction_output_md])

        profile_dd_advice.change(fn=lambda x:gr.update(value=x), inputs=profile_dd_advice, outputs=profile_in_tb)

        gen_adv_btn.click(
            fn=generate_advice_for_models_tab,
            inputs=[profile_in_tb, model_cbg_advice],
            outputs=[adv_disp_md, main_tabs, generated_advice_text_state, generated_evaluation_text_state, content_for_saving_state]
        ).then(
            lambda advice, eval_text: (advice + "\n\n--- değerlendirme ---\n\n" + eval_text) if advice and eval_text else advice,
            inputs=[generated_advice_text_state, generated_evaluation_text_state],
            outputs=[content_to_save_display_md]
        )

        eval_advice_btn.click(
            fn=evaluate_current_advice_with_llm,
            inputs=[generated_advice_text_state, llm_for_eval_dd_advise],
            outputs=[advice_evaluation_output_md, generated_evaluation_text_state, content_for_saving_state]
        ).then(
            lambda advice, eval_text: (advice + "\n\n--- DEĞERLENDİRME ---\n\n" + eval_text) if advice and eval_text else (advice or eval_text or ""),
            inputs=[generated_advice_text_state, generated_evaluation_text_state],
            outputs=[content_to_save_display_md]
        )

        @main_tabs.select
        def update_save_tab_preview(selected_tab_id: gr.SelectData):
            if selected_tab_id.index == 7:
                adv = generated_advice_text_state.value if generated_advice_text_state.value else ""
                evl = generated_evaluation_text_state.value if generated_evaluation_text_state.value else ""
                if adv and evl:
                    combined = adv + "\n\n" + evl
                elif adv:
                    combined = adv
                elif evl:
                    combined = evl
                else:
                    combined = "No content yet. Please generate advice and/or evaluation in Tab 7."
                return gr.update(value=combined)
            return gr.update()

        save_all_btn.click(fn=save_advice_and_evaluation_tab, inputs=[save_format_cbg, content_for_saving_state], outputs=[dl_final_file, save_final_status_md])

        def initial_app_load_actions():
            initial_target = current_target_disease
            advice_tab_text = f"**Advice & Evaluation will relate to: `{initial_target}`** (if rules are extracted for this target)." if initial_target else "**Advice Target: `No Target Selected in Tab 3`**."
            return initial_target, advice_tab_text

        demo.load(
            fn=initial_app_load_actions,
            inputs=None,
            outputs=[current_target_disease_state, current_advice_target_md]
        )

    return demo

if __name__ == "__main__":
    app = create_gradio_app()
    app.launch(debug=True, share=True)