"""
Mock data generators for development and testing purposes (PHASE 10)
"""
import random
import string
import uuid
from typing import Dict, Any, List
from datetime import datetime


class MockDataGenerator:
    """Generator for realistic mock pipeline data"""

    @staticmethod
    def generate_job_id() -> str:
        """Generate a realistic job ID"""
        return str(uuid.uuid4())

    @staticmethod
    def generate_input_text(length: int = 150) -> str:
        """Generate random input text"""
        questions = [
            "What is the capital of France?",
            "Who invented the telephone?",
            "What is the largest planet in our solar system?",
            "How many continents are there?",
            "What is the chemical symbol for gold?",
            "In what year did World War II end?",
            "Who wrote Romeo and Juliet?",
            "What is the smallest prime number?",
            "Which country has the largest population?",
            "What is the speed of light?",
        ]
        return random.choice(questions)

    @staticmethod
    def generate_hidden_states() -> Dict[str, Any]:
        """Generate mock hidden state outputs from LLM"""
        return {
            "token_embeddings": [round(random.random(), 2) for _ in range(10)],
            "attention_maps": [round(random.random(), 2) for _ in range(8)],
            "layer_outputs": [round(random.random(), 2) for _ in range(5)],
        }

    @staticmethod
    def generate_extracted_features() -> Dict[str, Any]:
        """Generate mock feature extraction results"""
        return {
            "dynamic_layer_sampling": [round(random.random(), 2) for _ in range(3)],
            "multi_scale_attention": [round(random.random(), 2) for _ in range(3)],
            "transformer_encoder": [round(random.random(), 2) for _ in range(3)],
            "self_attention_pooling": [round(random.random(), 2) for _ in range(3)],
        }

    @staticmethod
    def generate_hallucination_result(is_hallucination: bool = None) -> Dict[str, Any]:
        """Generate mock hallucination detection result"""
        if is_hallucination is None:
            is_hallucination = random.choice([True, False])

        probability = 0.95 if is_hallucination else 0.15
        probability += random.uniform(-0.1, 0.1)
        probability = max(0, min(1, probability))

        return {
            "prediction": is_hallucination,
            "probability": round(probability, 3),
            "confidence": round(probability * 100, 2),
            "model_votes": {
                "random_forest": is_hallucination,
                "xgboost": is_hallucination,
                "lightgbm": is_hallucination,
                "logistic_regression": is_hallucination or random.choice([True, False]),
                "svm": is_hallucination,
            },
            "stacking": "out_of_fold",
        }

    @staticmethod
    def generate_retrieved_evidence() -> Dict[str, Any]:
        """Generate mock RAG verification evidence"""
        sources = [
            "Wikipedia",
            "FEVER Dataset",
            "Common Sense Knowledge Base",
            "Academic Papers",
        ]
        return {
            "sources": random.sample(sources, k=random.randint(1, 3)),
            "supporting_documents": [
                f"Document about topic {i}" for i in range(random.randint(1, 3))
            ],
            "contradictions": (
                ["Contradicting statement found"]
                if random.choice([True, False])
                else []
            ),
        }

    @staticmethod
    def generate_explanation() -> Dict[str, Any]:
        """Generate mock explainability results"""
        return {
            "shap_values": [round(random.random(), 2) for _ in range(5)],
            "important_tokens": ["word" + str(i) for i in range(random.randint(3, 7))],
            "attention_heatmap": f"generated/heatmap_{uuid.uuid4()}.png",
            "explanation_text": "The model prediction was based on token importance analysis and evidence retrieval.",
        }

    @staticmethod
    def generate_full_pipeline_state(
        job_id: str = None, is_hallucination: bool = None
    ) -> Dict[str, Any]:
        """Generate complete mock pipeline state"""
        if job_id is None:
            job_id = MockDataGenerator.generate_job_id()

        return {
            "job_id": job_id,
            "input_text": MockDataGenerator.generate_input_text(),
            "input_type": "text",
            "generated_response": f"Generated response for {job_id}",
            "hidden_states": MockDataGenerator.generate_hidden_states(),
            "extracted_features": MockDataGenerator.generate_extracted_features(),
            "hallucination_result": MockDataGenerator.generate_hallucination_result(
                is_hallucination
            ),
            "retrieved_evidence": MockDataGenerator.generate_retrieved_evidence(),
            "explanation": MockDataGenerator.generate_explanation(),
        }

    @staticmethod
    def generate_stage_events(job_id: str = None, num_stages: int = 8) -> List[Dict[str, Any]]:
        """Generate mock stage progress events"""
        if job_id is None:
            job_id = MockDataGenerator.generate_job_id()

        stage_names = [
            "Input Received",
            "Generating Response",
            "Hidden State Extraction",
            "Feature Extraction",
            "Hallucination Detection",
            "Fact Verification",
            "Explainability",
            "Analysis Completed",
        ]

        events = []
        for i in range(1, num_stages + 1):
            progress = int((i / num_stages) * 100)
            events.append(
                {
                    "job_id": job_id,
                    "stage": i,
                    "name": stage_names[i - 1],
                    "status": "completed",
                    "progress_percentage": progress,
                    "duration_ms": random.randint(500, 2000),
                    "metadata": (
                        MockDataGenerator.generate_hidden_states()
                        if i == 3
                        else MockDataGenerator.generate_extracted_features()
                        if i == 4
                        else {}
                    ),
                }
            )
        return events

    @staticmethod
    def generate_batch_analysis_results(
        num_jobs: int = 10,
    ) -> List[Dict[str, Any]]:
        """Generate multiple mock analysis results for load testing"""
        results = []
        for _ in range(num_jobs):
            is_hallucination = random.choice([True, False])
            results.append(
                {
                    "job_id": MockDataGenerator.generate_job_id(),
                    "input_text": MockDataGenerator.generate_input_text(),
                    "hallucination": is_hallucination,
                    "confidence": round(
                        0.95 if is_hallucination else 0.15, 2
                    ),
                    "processing_time_ms": random.randint(3000, 5000),
                    "status": "completed",
                }
            )
        return results
