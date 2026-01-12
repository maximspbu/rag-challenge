import json
import requests
import sys
from typing import List, Dict, Any
from tqdm import tqdm

from src.config import QUESTIONS_PATH, OUTPUT_FILE, TEAM_EMAIL, SUBMISSION_NAME, SUBMISSION_URL
from src.retrieval import load_resources, create_retriever_pipeline
from src.generation import build_retrieve_fn, create_rag_chain


def run_pipeline() -> None:
    """
    Executes the inference pipeline.
    """
    # 1. Load Data
    try:
        vector_db, documents, known_companies = load_resources()
    except Exception as e:
        print(f"Critical Error loading resources: {e}")
        print("Did you run --ingest and --index?")
        sys.exit(1)

    bm25, v_db, compressor = create_retriever_pipeline(vector_db, documents)

    # 2. Build Chain
    retrieve_fn = build_retrieve_fn(v_db, bm25, compressor, known_companies)
    rag_chain = create_rag_chain(retrieve_fn)

    # 3. Load Questions
    print(f"Loading questions from {QUESTIONS_PATH}")
    if not QUESTIONS_PATH.exists():
        print(f"Error: {QUESTIONS_PATH} not found.")
        return

    with open(QUESTIONS_PATH, "r") as f:
        questions: List[Dict[str, Any]] = json.load(f)

    # 4. Inference
    answers_list: List[Dict[str, Any]] = []

    for i, item in enumerate(tqdm(questions, desc="Processing Questions")):
        q_text = item.get("text", "")
        q_kind = item.get("kind", "")

        input_data = {"question": q_text, "kind": q_kind}

        try:
            result = rag_chain.invoke(input_data)
            ans_dict = result.model_dump()
            ans_dict["question_text"] = q_text
            ans_dict["kind"] = q_kind
            answers_list.append(ans_dict)
        except Exception as e:
            print(f"Error processing question {i}: {e}")
            answers_list.append({"question_text": q_text, "kind": q_kind, "value": "N/A", "references": []})

    # 5. Save & Submit
    submission = {"team_email": TEAM_EMAIL, "submission_name": SUBMISSION_NAME, "answers": answers_list}

    with open(OUTPUT_FILE, "w") as f:
        json.dump(submission, f, indent=2, default=str)

    print(f"Submission saved to {OUTPUT_FILE}")

    if SUBMISSION_URL:
        print(f"Submitting to {SUBMISSION_URL}...")
        try:
            with open(OUTPUT_FILE, "rb") as f:
                files = {"file": (str(OUTPUT_FILE), f, "application/json")}
                headers = {"accept": "application/json"}
                resp = requests.post(SUBMISSION_URL, headers=headers, files=files)
                print("Submission Response:", resp.json())
        except Exception as e:
            print(f"Submission failed: {e}")
