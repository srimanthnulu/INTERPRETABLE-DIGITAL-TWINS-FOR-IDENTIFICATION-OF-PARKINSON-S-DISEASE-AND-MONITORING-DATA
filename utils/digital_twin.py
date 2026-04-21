"""
Digital Twin System for Patient Monitoring
Tracks patient predictions and disease progression over time
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, asdict
import pickle


@dataclass
class PredictionRecord:
    """Single prediction record"""
    timestamp: str
    image_type: str
    prediction: int
    probability: float
    confidence: float
    attention_map_path: str

    def to_dict(self):
        return asdict(self)


@dataclass
class PatientProfile:
    """Patient profile with medical history"""
    patient_id: str
    name: str
    age: int
    gender: str
    medical_history: str
    created_at: str

    def to_dict(self):
        return asdict(self)


class DigitalTwin:
    """Digital Twin for Parkinson's Disease monitoring"""

    def __init__(self, storage_dir: str = 'patient_records'):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)

    def create_patient(self, patient_id: str, name: str, age: int,
                      gender: str, medical_history: str = "") -> PatientProfile:
        profile = PatientProfile(
            patient_id=patient_id,
            name=name,
            age=age,
            gender=gender,
            medical_history=medical_history,
            created_at=datetime.now().isoformat()
        )
        patient_dir = os.path.join(self.storage_dir, patient_id)
        os.makedirs(patient_dir, exist_ok=True)
        profile_path = os.path.join(patient_dir, 'profile.json')
        with open(profile_path, 'w') as f:
            json.dump(profile.to_dict(), f, indent=4)
        predictions_path = os.path.join(patient_dir, 'predictions.json')
        with open(predictions_path, 'w') as f:
            json.dump([], f)
        return profile

    def add_prediction(self, patient_id: str, image_type: str,
                      prediction: int, probability: float,
                      attention_map_path: str = "") -> PredictionRecord:
        record = PredictionRecord(
            timestamp=datetime.now().isoformat(),
            image_type=image_type,
            prediction=prediction,
            probability=probability,
            confidence=max(probability, 1 - probability),
            attention_map_path=attention_map_path
        )
        predictions_path = os.path.join(self.storage_dir, patient_id, 'predictions.json')
        if os.path.exists(predictions_path):
            with open(predictions_path, 'r') as f:
                predictions = json.load(f)
        else:
            predictions = []
        predictions.append(record.to_dict())
        with open(predictions_path, 'w') as f:
            json.dump(predictions, f, indent=4)
        return record

    def get_patient_history(self, patient_id: str) -> List[Dict]:
        predictions_path = os.path.join(self.storage_dir, patient_id, 'predictions.json')
        if not os.path.exists(predictions_path):
            return []
        with open(predictions_path, 'r') as f:
            predictions = json.load(f)
        return predictions

    def get_patient_profile(self, patient_id: str) -> Optional[Dict]:
        profile_path = os.path.join(self.storage_dir, patient_id, 'profile.json')
        if not os.path.exists(profile_path):
            return None
        with open(profile_path, 'r') as f:
            profile = json.load(f)
        return profile

    def analyze_progression(self, patient_id: str, save_path: str = None) -> Dict:
        history = self.get_patient_history(patient_id)
        if not history:
            return {'error': 'No prediction history found'}
        df = pd.DataFrame(history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        stats = {
            'total_predictions': len(df),
            'pd_predictions': (df['prediction'] == 1).sum(),
            'healthy_predictions': (df['prediction'] == 0).sum(),
            'average_probability': df['probability'].mean(),
            'average_confidence': df['confidence'].mean(),
            'latest_prediction': df.iloc[-1]['prediction'],
            'latest_probability': df.iloc[-1]['probability'],
            'risk_trend': self._calculate_trend(df['probability'].values)
        }
        if save_path:
            self._plot_progression(df, patient_id, save_path)
        return stats

    def _calculate_trend(self, probabilities: np.ndarray) -> str:
        if len(probabilities) < 2:
            return 'insufficient_data'
        x = np.arange(len(probabilities))
        slope = np.polyfit(x, probabilities, 1)[0]
        if slope > 0.05:
            return 'increasing'
        elif slope < -0.05:
            return 'decreasing'
        else:
            return 'stable'

    def _plot_progression(self, df: pd.DataFrame, patient_id: str, save_path: str):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        ax1.plot(df['timestamp'], df['probability'],
                marker='o', linestyle='-', linewidth=2, markersize=6)
        ax1.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Decision Threshold')
        ax1.fill_between(df['timestamp'], 0, df['probability'], alpha=0.3, label='PD Risk')
        ax1.set_xlabel('Date')
        ax1.set_ylabel("Parkinson's Probability")
        ax1.set_title(f'Disease Progression for Patient {patient_id}')
        ax1.legend()
        ax1.grid(alpha=0.3)
        ax1.set_ylim([0, 1])
        image_types = df['image_type'].value_counts()
        ax2.bar(image_types.index, image_types.values, color='steelblue')
        ax2.set_xlabel('Image Type')
        ax2.set_ylabel('Number of Tests')
        ax2.set_title('Test Distribution by Image Type')
        ax2.grid(alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def generate_report(self, patient_id: str, save_path: str = None) -> str:
        profile = self.get_patient_profile(patient_id)
        history = self.get_patient_history(patient_id)
        stats = self.analyze_progression(patient_id)

        # Pre-compute values with quotes/backslashes to avoid f-string issues
        separator_thick = '=' * 80
        separator_thin = '-' * 80
        separator_mid = '-' * 40
        medical_history_text = profile['medical_history'] if profile['medical_history'] else 'No medical history recorded'
        latest_status = "PARKINSON'S DISEASE DETECTED" if stats.get('latest_prediction') == 1 else "HEALTHY"
        avg_prob = f"{stats.get('average_probability', 0):.2%}"
        avg_conf = f"{stats.get('average_confidence', 0):.2%}"
        latest_prob = f"{stats.get('latest_probability', 0):.2%}"
        risk_trend = stats.get('risk_trend', 'N/A').upper()

        report = f"""{separator_thick}
DIGITAL TWIN - PARKINSON'S DISEASE MONITORING REPORT
{separator_thick}

PATIENT INFORMATION
{separator_thin}
Patient ID:       {profile['patient_id']}
Name:             {profile['name']}
Age:              {profile['age']} years
Gender:           {profile['gender']}
Profile Created:  {profile['created_at'][:10]}

MEDICAL HISTORY
{separator_thin}
{medical_history_text}

TESTING SUMMARY
{separator_thin}
Total Tests:           {stats.get('total_predictions', 0)}
Parkinson's Detected:  {stats.get('pd_predictions', 0)}
Healthy Results:       {stats.get('healthy_predictions', 0)}
Average Probability:   {avg_prob}
Average Confidence:    {avg_conf}

LATEST ASSESSMENT
{separator_thin}
Prediction:      {latest_status}
Risk Score:      {latest_prob}
Risk Trend:      {risk_trend}

RECENT TEST HISTORY
{separator_thin}
"""

        for record in history[-5:]:
            result_label = "PD" if record['prediction'] == 1 else "Healthy"
            rec_prob = f"{record['probability']:.2%}"
            rec_conf = f"{record['confidence']:.2%}"
            report += f"""
Date:        {record['timestamp'][:10]}
Image Type:  {record['image_type']}
Result:      {result_label}
Probability: {rec_prob}
Confidence:  {rec_conf}
{separator_mid}
"""

        report += f"\n{separator_thick}\n"

        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)

        return report

    def list_patients(self) -> List[str]:
        if not os.path.exists(self.storage_dir):
            return []
        patients = [d for d in os.listdir(self.storage_dir)
                   if os.path.isdir(os.path.join(self.storage_dir, d))]
        return patients