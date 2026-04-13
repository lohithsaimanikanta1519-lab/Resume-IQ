"""
ResumeIQ — Flask Backend
========================
Upload resumes (PDF or TXT) + paste a job description → get ranked results.
"""

import os, re, json
from collections import defaultdict
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Config ────────────────────────────────────────────────────────────────────
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
ALLOWED_EXT   = {"pdf", "txt"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024   # 10 MB per file

# ── Skill Database ────────────────────────────────────────────────────────────
SKILLS_DB = {
    "Programming Languages": [
        "python","java","javascript","typescript","c++","c#","ruby","go",
        "rust","kotlin","swift","php","scala","r","matlab","perl","bash","shell",
    ],
    "Web Technologies": [
        "html","css","react","angular","vue","nodejs","django","flask","fastapi",
        "spring","express","rest","graphql","webpack","nextjs","nuxtjs","tailwind",
    ],
    "Data & ML": [
        "machine learning","deep learning","nlp","natural language processing",
        "tensorflow","pytorch","keras","scikit-learn","pandas","numpy","matplotlib",
        "seaborn","xgboost","lightgbm","computer vision","data analysis",
        "feature engineering","model deployment","transformers","bert","llm",
        "data science","statistics","regression","classification","clustering",
    ],
    "Cloud & DevOps": [
        "aws","azure","gcp","docker","kubernetes","terraform","ansible","jenkins",
        "ci/cd","linux","git","github","gitlab","devops","mlops","airflow","spark",
        "hadoop","kafka","rabbitmq",
    ],
    "Databases": [
        "sql","mysql","postgresql","mongodb","redis","elasticsearch","sqlite",
        "oracle","nosql","cassandra","dynamodb","firebase","supabase",
    ],
    "Soft Skills": [
        "communication","teamwork","leadership","problem solving","project management",
        "agile","scrum","time management","collaboration","critical thinking",
        "presentation","mentoring","research",
    ],
}
ALL_SKILLS = {s for skills in SKILLS_DB.values() for s in skills}

STOPWORDS = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with","by",
    "from","as","is","was","are","were","be","been","being","have","has","had",
    "do","does","did","will","would","could","should","may","might","shall","can",
    "that","this","these","those","i","we","you","he","she","it","they","me","us",
    "him","her","them","my","our","your","his","their","its","what","which","who",
    "how","all","each","every","both","few","more","most","other","some","such",
    "than","too","very","just","also","into","over","after","about","above","below",
    "between","through","during","then","now","when","where","why","so","if","not",
}

# ── Text utilities ────────────────────────────────────────────────────────────

def extract_text_from_file(filepath: str) -> str:
    ext = filepath.rsplit(".", 1)[-1].lower()
    if ext == "pdf":
        text = []
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text.append(t)
        return "\n".join(text)
    else:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s\+\#\/\.]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess(text: str) -> str:
    tokens = clean_text(text).split()
    return " ".join(t for t in tokens if t not in STOPWORDS and len(t) > 1)


def extract_skills(text: str) -> dict:
    text_lower = text.lower()
    found = defaultdict(list)
    for cat, skills in SKILLS_DB.items():
        for skill in skills:
            if re.search(r"\b" + re.escape(skill) + r"\b", text_lower):
                found[cat].append(skill)
    return dict(found)


def flat_skills(text: str) -> set:
    return {s for skills in extract_skills(text).values() for s in skills}


def extract_name(text: str, filename: str) -> str:
    """Best-effort: first non-empty line that looks like a name."""
    for line in text.splitlines():
        line = line.strip()
        if 3 < len(line) < 50 and not any(c.isdigit() for c in line) \
                and not any(kw in line.lower() for kw in ["resume","cv","curriculum","@","http"]):
            words = line.split()
            if 1 < len(words) <= 5:
                return line.title()
    # Fall back to filename
    return os.path.splitext(filename)[0].replace("_", " ").replace("-", " ").title()


# ── Core screening logic ──────────────────────────────────────────────────────

def screen_resumes(job_description: str, resumes: list[dict],
                   tfidf_w: float = 0.6, skill_w: float = 0.4) -> list[dict]:
    """
    resumes: [{"name": str, "text": str, "filename": str}]
    Returns ranked list of result dicts.
    """
    jd_skills = flat_skills(job_description)
    jd_proc   = preprocess(job_description)

    all_texts = [jd_proc] + [preprocess(r["text"]) for r in resumes]
    vec       = TfidfVectorizer(ngram_range=(1, 2), max_features=8000)
    tfidf_mat = vec.fit_transform(all_texts)
    jd_vec    = tfidf_mat[0]

    results = []
    for i, r in enumerate(resumes):
        tfidf_score = float(cosine_similarity(jd_vec, tfidf_mat[i + 1])[0][0])
        res_skills  = flat_skills(r["text"])
        matched     = jd_skills & res_skills
        skill_score = len(matched) / len(jd_skills) if jd_skills else 0.0
        final_score = tfidf_w * tfidf_score + skill_w * skill_score

        results.append({
            "name":           r["name"],
            "filename":       r["filename"],
            "final_score":    round(final_score * 100, 2),
            "tfidf_score":    round(tfidf_score * 100, 2),
            "skill_score":    round(skill_score * 100, 2),
            "matched_skills": sorted(matched),
            "missing_skills": sorted(jd_skills - res_skills),
            "skill_breakdown": extract_skills(r["text"]),
            "total_required": len(jd_skills),
        })

    results.sort(key=lambda x: x["final_score"], reverse=True)
    for rank, r in enumerate(results, 1):
        r["rank"] = rank
    return results


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/screen", methods=["POST"])
def screen():
    job_description = request.form.get("job_description", "").strip()
    if not job_description:
        return jsonify({"error": "Job description is required."}), 400

    files = request.files.getlist("resumes")
    if not files or all(f.filename == "" for f in files):
        return jsonify({"error": "Please upload at least one resume."}), 400

    resumes = []
    errors  = []
    for f in files:
        if f.filename == "":
            continue
        ext = f.filename.rsplit(".", 1)[-1].lower() if "." in f.filename else ""
        if ext not in ALLOWED_EXT:
            errors.append(f"{f.filename}: unsupported format (use PDF or TXT)")
            continue
        fname    = secure_filename(f.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], fname)
        f.save(filepath)
        try:
            text = extract_text_from_file(filepath)
            if not text.strip():
                errors.append(f"{f.filename}: could not extract text")
                continue
            name = extract_name(text, f.filename)
            resumes.append({"name": name, "text": text, "filename": f.filename})
        except Exception as e:
            errors.append(f"{f.filename}: {str(e)}")
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

    if not resumes:
        return jsonify({"error": "No valid resumes could be processed. " + " | ".join(errors)}), 400

    try:
        results = screen_resumes(job_description, resumes)
    except Exception as e:
        return jsonify({"error": f"Screening failed: {str(e)}"}), 500

    return jsonify({"results": results, "warnings": errors, "jd_skills": sorted(flat_skills(job_description))})


if __name__ == "__main__":
    print("\n🚀  ResumeIQ is running...\n")
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
