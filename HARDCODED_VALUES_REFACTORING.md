# Hardcoded Values Refactoring Summary

## Overview
All hardcoded configuration values have been extracted and moved to `config.py` for centralized management and environment-based configuration.

## Changes Made

### 1. **config.py** - New Configuration Constants Added

#### Pipeline Stage Configuration
- `PIPELINE_STAGE_DEFINITIONS`: Replaced hardcoded 8-stage pipeline with configurable list
  - Format: `(stage_number, stage_name, progress_percentage)`
  - Stages: Input Received (10%), Generating Response (20%), Hidden State Extraction (35%), Feature Extraction (50%), Hallucination Detection (70%), Fact Verification (85%), Explainability (95%), Analysis Completed (100%)

#### ML Model Thresholds
- `HALLUCINATION_PROBABILITY_THRESHOLD`: 0.5 (confidence threshold for hallucination detection)
- `BASE_HALLUCINATION_PROBABILITY`: 0.08 (base probability for unmatched content)
- `MATCH_PROBABILITY`: 0.12 (probability when response matches verified answer)
- `MISMATCH_PROBABILITY`: 0.86 (probability when response contradicts verified answer)
- `NO_ANSWER_PROBABILITY`: 0.42 (probability when no verified answer exists)

#### Feature Extraction Configuration
- `MAX_HIDDEN_STATE_TOKENS`: 4 (limit for token embeddings and attention maps)
- `MAX_ATTENTION_TOKENS`: 4 (limit for attention maps)
- `MAX_MULTI_SCALE_ATTENTION_TOKENS`: 3 (limit for multi-scale attention)
- `MAX_TRANSFORMER_ENCODER_FEATURES`: 3 (number of transformer encoder features)
- `SHAP_EXPLANATION_STEPS`: 3 (number of SHAP explanation steps)

#### Rate Limiting Configuration
- `RATE_LIMIT_WINDOW_SECONDS`: 60 (time window for rate limiting)

#### Logging Configuration
- `LOGS_DIRECTORY`: "logs" (log file directory)
- `LOG_FILE_PREFIX`: "hallucination_guard" (prefix for log files)
- `LOG_DATE_FORMAT`: "%Y-%m-%d %H:%M:%S" (datetime format in logs)
- `LOG_MESSAGE_FORMAT`: "%(asctime)s - %(name)s - %(levelname)s - %(message)s" (log message format)
- `LOG_FILE_DATE_FORMAT`: "%Y%m%d" (date format in log filenames)
- `LOGGER_NAME`: "hallucination_guard" (logger instance name)

#### Retry Configuration
- `RETRY_BACKOFF_BASE`: 0.5 (base multiplier for exponential backoff)
- `RETRY_BACKOFF_EXPONENT`: 2 (exponent for exponential backoff calculation)

#### Mock Data Configuration (Dev/Test)
- `MOCK_HALLUCINATION_PROBABILITY`: 0.95 (base probability for hallucination cases)
- `MOCK_NO_HALLUCINATION_PROBABILITY`: 0.15 (base probability for non-hallucination cases)
- `MOCK_PROBABILITY_VARIANCE`: 0.1 (variance range for mock data)

### 2. **services/pipeline_service.py** - Updated to Use Config Values

#### Changes:
- Removed hardcoded `STAGES` list from class definition
- Made `STAGES` instance variable loaded from `settings.PIPELINE_STAGE_DEFINITIONS`
- Updated `execute()` method to dynamically loop through configured stages instead of explicit calls
- Updated `_stage_3()`: Use `settings.MAX_HIDDEN_STATE_TOKENS` instead of `min(4, ...)`
- Updated `_stage_4()`: Use `settings.MAX_MULTI_SCALE_ATTENTION_TOKENS` and `settings.MAX_TRANSFORMER_ENCODER_FEATURES`
- Updated `_stage_5()`: Use configurable probability thresholds:
  - `settings.BASE_HALLUCINATION_PROBABILITY`
  - `settings.MATCH_PROBABILITY`
  - `settings.MISMATCH_PROBABILITY`
  - `settings.NO_ANSWER_PROBABILITY`
  - `settings.HALLUCINATION_PROBABILITY_THRESHOLD`
- Updated `_stage_7()`: Use `settings.SHAP_EXPLANATION_STEPS` instead of hardcoded 3
- Updated retry backoff: Use `settings.RETRY_BACKOFF_BASE` and `settings.RETRY_BACKOFF_EXPONENT`

### 3. **middleware/rate_limiter.py** - Updated to Use Config

#### Changes:
- Replaced `self.window_seconds = 60` with `self.window_seconds = settings.RATE_LIMIT_WINDOW_SECONDS`

### 4. **utils/logging_config.py** - Updated to Use Config

#### Changes:
- Import `settings` at module level
- Replace hardcoded "logs" directory with `settings.LOGS_DIRECTORY`
- Replace hardcoded "hallucination_guard" prefix with `settings.LOG_FILE_PREFIX`
- Replace hardcoded date format "%Y%m%d" with `settings.LOG_FILE_DATE_FORMAT`
- Replace hardcoded logger name with `settings.LOGGER_NAME` (function parameter default)
- Replace hardcoded log message format with `settings.LOG_MESSAGE_FORMAT`
- Replace hardcoded date format "%Y-%m-%d %H:%M:%S" with `settings.LOG_DATE_FORMAT`

### 5. **utils/mock_data.py** - Updated to Use Config

#### Changes:
- Added `from config import settings` import
- Updated `generate_hallucination_result()` to use:
  - `settings.MOCK_HALLUCINATION_PROBABILITY`
  - `settings.MOCK_NO_HALLUCINATION_PROBABILITY`
  - `settings.MOCK_PROBABILITY_VARIANCE`

## Environment Configuration

All configuration values can now be overridden via environment variables:

```bash
# Pipeline Configuration
export PIPELINE_STAGE_DEFINITIONS="1,Input Received,10;2,Generating Response,20;..."

# ML Thresholds
export HALLUCINATION_PROBABILITY_THRESHOLD=0.5
export BASE_HALLUCINATION_PROBABILITY=0.08
export MATCH_PROBABILITY=0.12
export MISMATCH_PROBABILITY=0.86
export NO_ANSWER_PROBABILITY=0.42

# Features
export MAX_HIDDEN_STATE_TOKENS=4
export MAX_MULTI_SCALE_ATTENTION_TOKENS=3
export MAX_TRANSFORMER_ENCODER_FEATURES=3
export SHAP_EXPLANATION_STEPS=3

# Rate Limiting
export RATE_LIMIT_WINDOW_SECONDS=60

# Logging
export LOGS_DIRECTORY=logs
export LOG_FILE_PREFIX=hallucination_guard
export LOGGER_NAME=hallucination_guard
export LOG_DATE_FORMAT="%Y-%m-%d %H:%M:%S"
export LOG_FILE_DATE_FORMAT="%Y%m%d"

# Retry Configuration
export RETRY_BACKOFF_BASE=0.5
export RETRY_BACKOFF_EXPONENT=2

# Mock Data
export MOCK_HALLUCINATION_PROBABILITY=0.95
export MOCK_NO_HALLUCINATION_PROBABILITY=0.15
export MOCK_PROBABILITY_VARIANCE=0.1
```

## Benefits

1. **Centralized Configuration**: All values in one place (`config.py`)
2. **Environment-Based**: Easy to customize per environment (dev, test, prod)
3. **Maintainability**: Changes to thresholds/parameters don't require code changes
4. **Consistency**: Uniform pattern throughout the codebase
5. **Testing**: Mock values are now configurable for different test scenarios
6. **Documentation**: Configuration values are self-documenting

## Files Modified

- ✅ `config.py` - Added 30+ configuration constants
- ✅ `services/pipeline_service.py` - Refactored to use config values
- ✅ `middleware/rate_limiter.py` - Updated rate limiting configuration
- ✅ `utils/logging_config.py` - Removed hardcoded logging parameters
- ✅ `utils/mock_data.py` - Updated mock data generation

## Verification

All modified files have been validated for Python syntax and compile successfully.
