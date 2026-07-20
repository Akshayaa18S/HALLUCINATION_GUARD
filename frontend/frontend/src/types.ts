export type InputMode = "text" | "image" | "both";

export type StageStatus = "pending" | "running" | "completed" | "failed";

export interface StageDef {
  number: number;
  name: string;
  description: string;
}

export interface StageState extends StageDef {
  status: StageStatus;
  durationMs?: number;
}

export interface ModelVotes {
  random_forest?: boolean;
  xgboost?: boolean;
  lightgbm?: boolean;
  logistic_regression?: boolean;
  svm?: boolean;
}

export interface RetrievedEvidence {
  sources?: string[];
  supporting_documents?: string[];
  evidence?: string[];
  contradictions?: string[];
}

export interface AnalysisResult {
  job_id: string;
  user_query?: string;
  generated_response: string;
  verified_answer: string;
  hallucination: boolean;
  confidence: number; // 0-1
  hallucination_probability?: number;
  model_votes?: ModelVotes;
  retrieved_evidence: RetrievedEvidence;
  explanation: string;
  processing_time_ms?: number;
  tokens_input?: number;
  tokens_output?: number;
  created_at?: string;
}

export const PIPELINE_STAGES: StageDef[] = [
  { number: 1, name: "Input Received", description: "User prompt successfully received" },
  { number: 2, name: "Generating Response", description: "Generating response from the language model" },
  { number: 3, name: "Extracting Hidden States", description: "Extracting hidden states from selected layers" },
  { number: 4, name: "Feature Extraction", description: "Applying Multi-Scale Attention and Transformer" },
  { number: 5, name: "Hallucination Detection (Ensemble)", description: "Running ensemble models for prediction" },
  { number: 6, name: "RAG Verification", description: "Retrieving evidence from Wikipedia and FEVER" },
  { number: 7, name: "Explainability Generation", description: "Generating SHAP explanations and insights" },
  { number: 8, name: "Analysis Completed", description: "All steps completed successfully" },
];
