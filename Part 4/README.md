# Part 4 — LLM Powered Feature  
## Track B: Tabular Record Batch Scoring

---

# 1. Project Overview

This project implements an LLM-powered structured scoring system using the Adult Income Dataset.

The goal is to build a feature that demonstrates:

- LLM API integration using Python requests
- Prompt engineering
- Structured JSON output generation
- JSON schema validation
- PII safety guardrails
- Batch scoring pipeline
- Temperature-based output comparison

---

# 2. Selected Track

## Track B — Tabular Record Batch Scoring

Track B is selected for this project.

Three records from the cleaned Adult Income dataset are formatted as JSON objects and sent individually to an LLM API.

The LLM evaluates each record using a business scoring rubric and returns a structured JSON assessment.

Each response is validated against a predefined JSON schema.

---

# 3. Dataset Description

## Dataset Name

Adult Income Dataset

## Target Variable

`income`

Classes:

- `<=50K`
- `>50K`

## Features Used

- age
- workclass
- fnlwgt
- education
- education_num
- marital_status
- occupation
- relationship
- race
- sex
- capital_gain
- capital_loss
- hours_per_week
- native_country

---

# 4. Repository Structure

```
LLM_Feature_Project/

│
├── cleaned_data.csv
│
├── llm_pipeline.py
│
├── requirements.txt
│
├── README.md
│
└── .env
```

---

# 5. Environment Setup

Install required libraries:

```bash
pip install requests python-dotenv jsonschema pandas
```

---

# 6. API Key Configuration

The LLM API key is stored using an environment variable.

Example `.env` file:

```
LLM_API_KEY=your_api_key_here
```

The API key is never hardcoded inside the Python code.

---

# 7. LLM API Connection

A reusable function `call_llm()` is implemented.

Function:

```python
call_llm(
    system_prompt,
    user_prompt,
    temperature=0.0,
    max_tokens=512
)
```

The function performs:

1. Creates JSON payload
2. Adds model and messages
3. Adds temperature and max_tokens
4. Sends HTTP POST request
5. Checks response status
6. Returns generated LLM content

---

# 8. LLM Test Call

Test Prompt:

```
Reply with only the word: hello
```

Expected Response:

```
hello
```

The successful response confirms that the API connection works correctly.

---

# 9. Prompt Design

## System Prompt

The LLM acts as an AI income prediction assistant.

Exact system prompt:

```
You are an AI income prediction assistant.

Your task is to analyze Adult Income Dataset records
and predict whether a person earns <=50K or >50K.

Instructions:
- Analyze the given person attributes.
- Generate an income assessment score between 0 and 100.
- Return ONLY valid JSON.
- Do not provide explanations outside JSON.
- Follow the provided JSON schema.

Evaluation criteria:
- Education level
- Work experience
- Occupation
- Working hours
- Capital gain and loss
- Employment type


Example Input:

{
"age":39,
"education":"Bachelors",
"workclass":"State-gov",
"occupation":"Adm-clerical",
"hours_per_week":40
}


Example Output:

{
"income_score":70,
"income_class":">50K",
"prediction":"High income probability",
"key_factors":"Education level and occupation indicate higher earning potential",
"recommendation":"Continue professional growth"
}
```

The system prompt contains exactly one worked input-output example.

---

# 10. User Prompt Template

Each dataset record is converted into JSON format.

Template:

```
Analyze the following Adult Income record.

Input Record:

{record_json}

Return only the JSON assessment.
```

---

# 11. JSON Schema Definition

The LLM output is validated using JSON schema.

```python
income_schema = {

"type":"object",

"required":[

"income_score",
"income_class",
"prediction",
"key_factors",
"recommendation"

],

"properties":{

"income_score":{
"type":"number"
},

"income_class":{
"type":"string"
},

"prediction":{
"type":"string"
},

"key_factors":{
"type":"string"
},

"recommendation":{
"type":"string"
}

}

}
```

The schema contains five required scalar fields.

---

# 12. Structured Output Handling

After every LLM response:

## Step 1

Remove extra spaces:

```python
response.strip()
```

## Step 2

Parse JSON:

```python
json.loads()
```

## Step 3

Validate:

```python
jsonschema.validate()
```

## Step 4

Handle failure:

If validation fails, return fallback output.

Fallback:

```json
{
"income_score":null,
"income_class":null,
"prediction":null,
"key_factors":null,
"recommendation":null
}
```

---

# 13. Batch Scoring Pipeline

Pipeline flow:

```
Dataset Record

        ↓

Convert Record to JSON

        ↓

PII Guardrail Check

        ↓

LLM API Call

        ↓

Strip Response

        ↓

JSON Parsing

        ↓

Schema Validation

        ↓

Validated Assessment
```

Three real records from `cleaned_data.csv` are processed.

---

# 14. PII Guardrail

Before every LLM call, input text is checked for personally identifiable information.

Detection includes:

- Email addresses
- Phone numbers


Regex:

```python
email_pattern =
r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'

phone_pattern =
r'\b\d{10}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b'
```

If PII is detected:

```
Input blocked: PII detected.
```

The LLM call is stopped.

---

# 15. Guardrail Test Results

| Input | Result |
|---|---|
| Contact: john.doe@example.com | Blocked |
| Age:39 Education:Bachelors | Allowed |

---

# 16. Temperature Selection

Temperature used:

```
temperature = 0
```

Reason:

Temperature 0 produces deterministic and consistent outputs.

For structured JSON generation tasks, predictable responses are important because the output must satisfy a fixed schema.

---

# 17. Temperature A/B Comparison

The three records were tested using:

- Temperature = 0
- Temperature = 0.7


| Input | Output at temp=0 | Output at temp=0.7 | Key Difference |
|---|---|---|---|
| Record 1 | Structured JSON | Structured JSON | Different wording |
| Record 2 | Structured JSON | Structured JSON | Different recommendation |
| Record 3 | Structured JSON | Structured JSON | Different explanation style |


Explanation:

At temperature=0, the model selects the highest probability token and produces stable outputs.

At temperature=0.7, the model samples from a wider probability distribution, producing more variation in responses.

---

# 18. Batch Scoring Results

| Input Record | LLM Assessment JSON | Validation Status |
|---|---|---|
| Record 1 | Generated JSON Assessment | PASS |
| Record 2 | Generated JSON Assessment | PASS |
| Record 3 | Generated JSON Assessment | PASS |

---

# 19. End-to-End Demonstration

| Input | LLM Output | Valid JSON | Pass/Block |
|---|---|---|---|
| Record 1 | JSON Assessment | PASS | PASS |
| Record 2 | JSON Assessment | PASS | PASS |
| Record 3 | JSON Assessment | PASS | PASS |

---

# 20. Acceptance Criteria Checklist

| Requirement | Status |
|---|---|
| call_llm implemented | ✅ |
| Test prompt returns visible response | ✅ |
| API key stored in environment variable | ✅ |
| Track B selected | ✅ |
| Three records formatted as JSON | ✅ |
| LLM scoring performed per record | ✅ |
| Few-shot system prompt included | ✅ |
| Exactly one worked example | ✅ |
| JSON schema with 5 required fields | ✅ |
| jsonschema.validate() used | ✅ |
| Validation errors handled | ✅ |
| Fallback response implemented | ✅ |
| PII guardrail implemented | ✅ |
| Email input blocked | ✅ |
| Clean input allowed | ✅ |
| Temperature=0 demonstrated | ✅ |
| Temperature A/B comparison included | ✅ |
| README tables included | ✅ |

---

# 21. Conclusion

This project demonstrates an end-to-end LLM-powered structured scoring system.

The implementation includes:

- Secure API integration
- Prompt engineering
- Structured JSON output
- Schema validation
- PII protection
- Batch scoring
- Temperature comparison

The system provides reliable and validated LLM-based assessments from structured tabular data.