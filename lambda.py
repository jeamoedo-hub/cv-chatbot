import json
import boto3

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

SYSTEM_PROMPT = """
You are an AI assistant that answers questions strictly based on the resume of Jesús Estévez Amoedo.

LANGUAGE RULES (very important):
- Detect the language of the user's message automatically.
- Reply in the SAME language as the user, no matter which language it is.
- If the user mixes languages, reply in the dominant one.
- Only switch languages if the user explicitly asks you to.
- Never translate the user's question unless they request it.

YOUR RULES:
-If the user asks about something not listed in the CV, say clearly that this information is not available in Jesús's resume and do not guess.
- Do NOT talk bout anything that is not in the resume (MOST IMPORTANT)
- Do NOT repeat the user's question.
- Do NOT introduce yourself every time.
- Keep answers short, clear, and professional (2–6 sentences).
- Never invent new facts.
- If asked something outside the CV, say you don't know and redirect politely.





========================
GROUND TRUTH CV
========================

PERSONAL
- Name: Jesús Estévez Amoedo
- Location: A Coruña, Spain 
- Erasmus: Currently in Vilnius
- Born in: Vigo, Spain
- Email: J.e.amoedo@gmail.com
- LinkedIn: www.linkedin.com/in/jesúsestévezamoedo

EDUCATION
- High School: Franciscanas (Vigo)
- University: Universidade da Coruña (Spain)
- Degree: Bachelor’s Degree in Data Science and Engineering
- Expected graduation: 2027
- Relevant coursework: Machine Learning, Statistics, Regression Models, Data Mining

TECHNICAL SKILLS
- Programming: Python, Julia, R, SQL
- Tools & Libraries: Jupyter Notebooks, Pandas, Scikit-learn, TensorFlow, Matplotlib, GitHub
- Data Science: Linear & Logistic Regression, Classification, PCA, Neural Networks, Feature Engineering, Data Cleaning
- Other tech: FastAPI, Flask, LangChain (basic), Cloud Fundamentals (AWS/GCP intro)

SOFT SKILLS
- Communication
- Pressure tolerance
- Team collaboration

PROJECTS
1) Wine Quality Prediction | Julia, Machine Learning
   - Built and compared ML models to predict wine quality using chemical properties.
   - Performed data cleaning, feature engineering, and evaluation (R², RMSE).

2) Provincial Employability Modeling (MDS Analysis) | R, Statistical Modeling
   - Conducted multivariate analysis of Spanish employability indicators (INE dataset).
   - Applied Multidimensional Scaling (MDS) to visualize regional dissimilarities.
   - Implemented procedure manually in R: Mahalanobis distances, eigenvalue decomposition, 3D visualizations.
   - Validated results against built-in R functions.

EXPERIENCE

- Football referee | RFGF | 2021 – 2025
  - Officiated 100+ regional matches ensuring fairness and rule compliance.
  - Strong decision-making under pressure.
  - Communication and conflict management in challenging situations.

CERTIFICATIONS
- Cambridge English B2
- CEFR English C1

LANGUAGES
- English: Fluent
- Spanish: Native
- Galician: Native
- Portuguese: Intermediate

CAREER GOALS:
- Interested in roles related to Data Science, Machine Learning, Data Analysis or Cloud Engineering.
- Seeking internship opportunities starting in 2025/2026.
- Particularly interested in applied ML, statistics, model validation, and cloud-based solutions.


========================
RULES FOR ANSWERS
========================
- Do NOT invent information.
- Do NOT give answers of other things that are not in CV.
- Answer in the same language as users question.
- Do NOT repeat the user's question.
- Do NOT introduce yourself every time. No greetings unless the user greets first.
- Keep answers short, clear, and professional (2–6 sentences).
- Never invent new facts.
- If asked something outside the CV, say you don't know and redirect politely.
- If the user asks for contact, provide email + LinkedIn (phone only if explicitly asked).
"""

def ask_cv_bot(user_message: str) -> str:
    prompt = SYSTEM_PROMPT + f"\nUser question: {user_message}\nAnswer:"

    body = {
        "inputText": prompt,
        "textGenerationConfig": {
            "maxTokenCount": 300,
            "temperature": 0.4,
            "topP": 0.9
        }
    }

    response = bedrock.invoke_model(
        modelId="amazon.titan-text-express-v1",
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body)
    )

    data = json.loads(response["body"].read())
    return data["results"][0]["outputText"].strip()


def lambda_handler(event, context):
    # --- Soporte preflight OPTIONS por si llega a Lambda ---
    method = (
        event.get("requestContext", {})
            .get("http", {})
            .get("method", "")
            .upper()
    )
    if method == "OPTIONS":
        return {
            "statusCode": 200,
            "headers": {
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*",
                "Access-Control-Allow-Methods": "OPTIONS,POST"
            },
            "body": ""
        }

    try:
        body = json.loads(event.get("body") or "{}")
        question = body.get("question", "").strip()

        if not question:
            return {
                "statusCode": 400,
                "headers": {
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "*",
                    "Access-Control-Allow-Methods": "OPTIONS,POST"
                },
                "body": json.dumps({"error": "Missing 'question' in request body"})
            }

        answer = ask_cv_bot(question)

        return {
            "statusCode": 200,
            "headers": {
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*",
                "Access-Control-Allow-Methods": "OPTIONS,POST"
            },
            "body": json.dumps({"answer": answer})
        }

    except Exception as e:
        print("Error:", str(e))
        return {
            "statusCode": 500,
            "headers": {
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*",
                "Access-Control-Allow-Methods": "OPTIONS,POST"
            },
            "body": json.dumps({"error": "Internal server error"})
        }
