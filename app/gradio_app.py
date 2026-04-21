"""
Gradio Web Application for Parkinson's Disease Detection
Interactive interface with Digital Twin monitoring
"""

import gradio as gr
import torch
import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
import plotly.graph_objects as go
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vit_model import ViTForParkinsons
from utils.data_preprocessing import preprocess_single_image
from utils.interpretability import ViTGradCAM, visualize_cam
from utils.digital_twin import DigitalTwin


class ParkinsonsApp:
    """Main application class"""
    
    def __init__(self, model_path: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.digital_twin = DigitalTwin(storage_dir='patient_records')
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def load_model(self, model_path: str):
        try:
            self.model = ViTForParkinsons(num_classes=2, pretrained=False)
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model = self.model.to(self.device)
            self.model.eval()
            return f"✓ Model loaded successfully from {model_path}"
        except Exception as e:
            return f"✗ Error loading model: {str(e)}"

    def predict_image(self, image, image_type, patient_id=None):
        if self.model is None:
            return "Error: No model loaded", None, None
        try:
            temp_path = 'temp_image.png'
            if isinstance(image, np.ndarray):
                Image.fromarray(image).save(temp_path)
            else:
                image.save(temp_path)
            image_tensor = preprocess_single_image(temp_path).to(self.device)
            with torch.no_grad():
                logits, _ = self.model(image_tensor, output_attentions=True)
                probabilities = torch.softmax(logits, dim=1)
                prediction = logits.argmax(dim=1).item()
                pd_prob = probabilities[0, 1].item()
            grad_cam = ViTGradCAM(self.model)
            cam = grad_cam.generate_cam(image_tensor, target_class=1)
            viz_path = 'temp_explanation.png'
            visualize_cam(temp_path, cam, save_path=viz_path)
            class_names = ['Healthy', "Parkinson's Disease"]
            predicted_class = class_names[prediction]
            healthy_prob = f"{probabilities[0, 0].item():.2%}"
            pd_prob_str = f"{pd_prob:.2%}"
            result_text = f"""
### Prediction Results

**Predicted Class:** {predicted_class}

**Confidence Scores:**
- Parkinson's Disease: {pd_prob_str}
- Healthy: {healthy_prob}

**Risk Assessment:**
{self._get_risk_assessment(pd_prob)}
"""
            prob_chart = self._create_probability_chart(probabilities[0].cpu().numpy())
            if patient_id and patient_id.strip():
                self.digital_twin.add_prediction(
                    patient_id=patient_id.strip(),
                    image_type=image_type,
                    prediction=prediction,
                    probability=pd_prob,
                    attention_map_path=viz_path
                )
                result_text += f"\n\n✓ Prediction recorded for patient: {patient_id}"
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return result_text, viz_path, prob_chart
        except Exception as e:
            return f"Error during prediction: {str(e)}", None, None

    def _get_risk_assessment(self, probability: float) -> str:
        if probability < 0.3:
            return "🟢 **Low Risk** - No significant indicators detected"
        elif probability < 0.6:
            return "🟡 **Moderate Risk** - Some indicators present, monitoring recommended"
        else:
            return "🔴 **High Risk** - Strong indicators detected, medical consultation recommended"

    def _create_probability_chart(self, probabilities):
        fig = go.Figure(data=[
            go.Bar(
                x=['Healthy', "Parkinson's Disease"],
                y=probabilities,
                marker_color=['green', 'red'],
                text=[f'{p:.1%}' for p in probabilities],
                textposition='auto',
            )
        ])
        fig.update_layout(
            title='Prediction Probabilities',
            yaxis_title='Probability',
            yaxis_range=[0, 1],
            height=300
        )
        return fig

    def create_patient(self, patient_id, name, age, gender, medical_history):
        try:
            if not patient_id or not name:
                return "Error: Patient ID and Name are required"
            self.digital_twin.create_patient(
                patient_id=patient_id.strip(),
                name=name.strip(),
                age=int(age) if age else 0,
                gender=gender,
                medical_history=medical_history.strip() if medical_history else ""
            )
            return f"✓ Patient profile created successfully for {name} (ID: {patient_id})"
        except Exception as e:
            return f"✗ Error creating patient: {str(e)}"

    def view_patient_history(self, patient_id):
        try:
            if not patient_id or not patient_id.strip():
                return "Please enter a Patient ID", None, None
            patient_id = patient_id.strip()
            profile = self.digital_twin.get_patient_profile(patient_id)
            if not profile:
                return f"Patient {patient_id} not found", None, None
            history = self.digital_twin.get_patient_history(patient_id)
            if not history:
                return f"No prediction history for patient {patient_id}", None, None
            progression_path = f'temp_progression_{patient_id}.png'
            stats = self.digital_twin.analyze_progression(patient_id, save_path=progression_path)
            status_text = "Parkinson's Detected" if stats['latest_prediction'] == 1 else "Healthy"
            risk_trend = stats['risk_trend'].upper()
            avg_prob = f"{stats['average_probability']:.2%}"
            latest_prob = f"{stats['latest_probability']:.2%}"
            summary = f"""
### Patient Information

**Name:** {profile['name']}
**Age:** {profile['age']} years
**Gender:** {profile['gender']}
**Patient ID:** {patient_id}

### Testing Summary

- **Total Tests:** {stats['total_predictions']}
- **PD Detections:** {stats['pd_predictions']}
- **Healthy Results:** {stats['healthy_predictions']}
- **Average PD Risk:** {avg_prob}
- **Risk Trend:** {risk_trend}

### Latest Assessment

- **Status:** {status_text}
- **Risk Score:** {latest_prob}
"""
            df = pd.DataFrame(history)
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
            df['prediction'] = df['prediction'].map({0: 'Healthy', 1: 'PD'})
            df['probability'] = df['probability'].apply(lambda x: f'{x:.2%}')
            df = df[['timestamp', 'image_type', 'prediction', 'probability']]
            return summary, progression_path, df
        except Exception as e:
            return f"Error: {str(e)}", None, None

    def generate_patient_report(self, patient_id):
        try:
            if not patient_id or not patient_id.strip():
                return "Please enter a Patient ID"
            report_path = f'patient_report_{patient_id.strip()}.txt'
            report = self.digital_twin.generate_report(
                patient_id.strip(),
                save_path=report_path
            )
            return report
        except Exception as e:
            return f"Error generating report: {str(e)}"


def create_interface():
    app = ParkinsonsApp()
    css = """
    .gradio-container {
        font-family: 'Arial', sans-serif;
    }
    .logo-container {
        text-align: center;
        padding: 20px;
    }
    """
    with gr.Blocks(css=css, title="Parkinson's Disease Detection System") as demo:
        gr.Markdown("""
        # 🧠 Parkinson's Disease Detection & Digital Twin System
        ### Advanced AI-Powered Diagnosis with Interpretable Results
        This system uses Vision Transformer (ViT) deep learning to detect Parkinson's Disease
        from spiral, wave, and handwriting images with explainable AI visualizations.
        """)
        with gr.Tabs():
            with gr.Tab("🔬 Image Analysis"):
                gr.Markdown("## Upload and Analyze Medical Images")
                with gr.Row():
                    with gr.Column():
                        model_path_input = gr.Textbox(
                            label="Model Path",
                            value="results/vit_best_model.pth",
                            placeholder="Path to trained model"
                        )
                        load_btn = gr.Button("Load Model", variant="primary")
                        model_status = gr.Textbox(label="Model Status", interactive=False)
                        gr.Markdown("---")
                        image_input = gr.Image(label="Upload Image", type="pil")
                        image_type = gr.Radio(
                            choices=["spiral", "wave", "handwriting"],
                            value="spiral",
                            label="Image Type"
                        )
                        patient_id_input = gr.Textbox(
                            label="Patient ID (optional)",
                            placeholder="Enter patient ID to track in Digital Twin"
                        )
                        predict_btn = gr.Button("🔍 Analyze Image", variant="primary", size="lg")
                    with gr.Column():
                        result_output = gr.Markdown(label="Results")
                        prob_chart = gr.Plot(label="Probability Distribution")
                        explanation_output = gr.Image(label="AI Explanation Heatmap")
                load_btn.click(fn=app.load_model, inputs=[model_path_input], outputs=[model_status])
                predict_btn.click(
                    fn=app.predict_image,
                    inputs=[image_input, image_type, patient_id_input],
                    outputs=[result_output, explanation_output, prob_chart]
                )
            with gr.Tab("👤 Patient Management"):
                gr.Markdown("## Create and Manage Patient Profiles")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Create New Patient")
                        new_patient_id = gr.Textbox(label="Patient ID*", placeholder="P001")
                        new_patient_name = gr.Textbox(label="Full Name*", placeholder="John Doe")
                        new_patient_age = gr.Number(label="Age", value=65)
                        new_patient_gender = gr.Radio(
                            choices=["Male", "Female", "Other"],
                            label="Gender",
                            value="Male"
                        )
                        new_patient_history = gr.Textbox(
                            label="Medical History",
                            placeholder="Enter relevant medical history...",
                            lines=4
                        )
                        create_btn = gr.Button("Create Patient Profile", variant="primary")
                        create_output = gr.Textbox(label="Status", interactive=False)
                    with gr.Column():
                        gr.Markdown("### Patient List")
                        patient_list = gr.Textbox(
                            label="Registered Patients",
                            value="\n".join(app.digital_twin.list_patients()),
                            lines=10,
                            interactive=False
                        )
                        refresh_btn = gr.Button("🔄 Refresh List")
                create_btn.click(
                    fn=app.create_patient,
                    inputs=[new_patient_id, new_patient_name, new_patient_age,
                            new_patient_gender, new_patient_history],
                    outputs=[create_output]
                )
                refresh_btn.click(
                    fn=lambda: "\n".join(app.digital_twin.list_patients()),
                    outputs=[patient_list]
                )
            with gr.Tab("📊 Digital Twin Monitoring"):
                gr.Markdown("## Track Patient History and Disease Progression")
                with gr.Row():
                    with gr.Column():
                        view_patient_id = gr.Textbox(
                            label="Patient ID",
                            placeholder="Enter Patient ID"
                        )
                        view_btn = gr.Button("View Patient History", variant="primary")
                        patient_summary = gr.Markdown(label="Patient Summary")
                    with gr.Column():
                        progression_plot = gr.Image(label="Disease Progression Chart")
                        history_table = gr.Dataframe(label="Test History")
                view_btn.click(
                    fn=app.view_patient_history,
                    inputs=[view_patient_id],
                    outputs=[patient_summary, progression_plot, history_table]
                )
            with gr.Tab("📄 Generate Reports"):
                gr.Markdown("## Comprehensive Patient Reports")
                report_patient_id = gr.Textbox(
                    label="Patient ID",
                    placeholder="Enter Patient ID"
                )
                generate_report_btn = gr.Button("Generate Report", variant="primary")
                report_output = gr.Textbox(
                    label="Patient Report",
                    lines=25,
                    interactive=False
                )
                generate_report_btn.click(
                    fn=app.generate_patient_report,
                    inputs=[report_patient_id],
                    outputs=[report_output]
                )
            with gr.Tab("ℹ️ About"):
                gr.Markdown("""
                ## About This System

                ### Features
                - **Vision Transformer (ViT) Model**: State-of-the-art deep learning architecture
                - **Transfer Learning**: Pre-trained on ImageNet, fine-tuned for Parkinson's detection
                - **Interpretable AI**: Grad-CAM and attention heatmaps show what the model focuses on
                - **Digital Twin**: Track patient history and disease progression over time
                - **Multi-Modal Analysis**: Supports spiral, wave, and handwriting images

                ### Model Architecture
                - Base Model: ViT-Base (google/vit-base-patch16-224)
                - Input Size: 224×224 RGB images
                - Fine-tuning: Last 2 transformer layers
                - Optimizer: AdamW with weight decay
                - Loss: Cross-entropy

                ### Evaluation Metrics
                The system provides comprehensive evaluation including:
                - Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix

                ### How to Use
                1. **Load Model**: Upload or specify path to trained model
                2. **Analyze Image**: Upload spiral/wave/handwriting image
                3. **View Results**: Get prediction with probability and explanation
                4. **Track Patients**: Create profiles and monitor progression
                5. **Generate Reports**: Comprehensive patient reports

                ### Citation
           Interpretable Digital Twin for Parkinson's Disease Identification
            Using Vision Transformer with Grad-CAM Explanations
---
                **Developed with ❤️ for medical AI research**
                """)
        gr.Markdown("""
        ---
        <div style='text-align: center; color: gray; font-size: 12px;'>
        ⚠️ This system is for research purposes only and should not replace professional medical diagnosis.
        </div>
        """)
    return demo


def main():
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True
    )


if __name__ == '__main__':
    main()