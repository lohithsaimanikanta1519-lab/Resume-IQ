# ResumeIQ — Resume Screening Web App

A full-stack web application that lets you **upload real resumes** (PDF/TXT), paste a job description, and instantly get ML-powered candidate rankings with skill gap analysis.

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install flask pdfplumber scikit-learn werkzeug
```

### 2. Run the App

```bash
python app.py
```

### 3. Open in Browser

```
http://127.0.0.1:5000
```

---

## 📁 Project Structure

```
resumeiq/
├── app.py              ← Flask backend (ML engine + API)
├── requirements.txt    ← Python dependencies
├── uploads/            ← Temp folder (auto-created, auto-cleaned)
└── templates/
    └── index.html      ← Full frontend (HTML + CSS + JS)
```

---

## 🧠 How It Works

### Upload Flow
1. User uploads PDF or TXT resumes + pastes job description
2. Frontend sends `multipart/form-data` POST to `/screen`
3. Backend extracts text from PDFs using `pdfplumber`
4. Text is cleaned and preprocessed

### Scoring Formula
```
Final Score = (0.6 × TF-IDF Cosine Similarity) + (0.4 × Skill Match Ratio)
```

| Component | Weight | What it measures |
|-----------|--------|-----------------|
| TF-IDF Similarity | 60% | Overall contextual relevance of resume to JD |
| Skill Match | 40% | Exact keyword coverage of required skills |

### Output (per candidate)
- Final Score, TF-IDF Score, Skill Match Score
- Matched skills list
- Missing skills list (skill gap)
- Skill breakdown by category
- Rank among all candidates

---

## 🎨 Features

- **Drag & drop** resume upload (PDF + TXT)
- **Multiple files** at once
- **Animated loading** steps
- **Interactive ranking table** with score bars
- **Click any candidate** → detailed modal with gauge chart
- **Skill coverage heatmap** across all candidates
- **Score comparison** bar chart

---

## 📋 API

### `POST /screen`

**Form Data:**
- `job_description` (string) — full JD text
- `resumes` (files) — one or more PDF/TXT files

**Response:**
```json
{
  "results": [
    {
      "rank": 1,
      "name": "Candidate Name",
      "filename": "resume.pdf",
      "final_score": 72.4,
      "tfidf_score": 65.2,
      "skill_score": 84.6,
      "matched_skills": ["python", "tensorflow", ...],
      "missing_skills": ["kubernetes", ...],
      "skill_breakdown": { "Data & ML": ["python", ...], ... },
      "total_required": 27
    }
  ],
  "jd_skills": ["python", "tensorflow", ...],
  "warnings": []
}
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python + Flask |
| PDF Parsing | pdfplumber |
| NLP / Vectorization | scikit-learn (TF-IDF) |
| Similarity | Cosine Similarity |
| Skill Extraction | Regex keyword matching |
| Frontend | Vanilla HTML + CSS + JS |
| Fonts | Google Fonts (Syne + DM Sans + DM Mono) |

👨‍💻 Author

Lohith Ayancha