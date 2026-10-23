# ♻️ ReWA.AI — Recycling-Reusable Waste Assistant

**Sort Today • Save Tomorrow**

ReWA.AI helps you answer three essential waste-sorting questions: **"What is this item? Which bin does it go in? How do I properly recycle or reuse it?"**

Take or upload a photo of a waste item. The app uses zero-shot vision AI (CLIP) to identify the item, provides official municipal segregation guidance (customized for Bengaluru's 2-Bin 1-Bag system), allows you to verify or correct the suggestion, and tracks sorting history in a Digital Twin virtual bin system with gamified **Virtual Eco Points**.

---

## 📸 Screenshots

*(To display your screenshots here, save PNG images into `docs/screenshots/`)*

- **Homepage & Core Sorting Flow:** `docs/screenshots/homepage.png` *(Add screenshot)*
- **AI Classification & Bengaluru Guidance:** `docs/screenshots/classification.png` *(Add screenshot)*
- **Virtual Digital Twin Bins:** `docs/screenshots/digital_twin.png` *(Add screenshot)*
- **Analytics & Human-AI Metrics:** `docs/screenshots/analytics.png` *(Add screenshot)*

---

## ✨ Features

| Feature | Description |
|---|---|
| 📷 **Photo Classification** | Upload photos, take live pictures with a webcam, or click "Try a sample photo". CLIP evaluates images across 22 candidate labels. |
| 📍 **Bengaluru Segregation Rules** | Tailored to Bengaluru's BBMP household waste system: **Wet Waste** (green bin), **Dry Waste** (blue bag), **Sanitary Waste** (red/newspaper), **E-Waste**, and **Domestic Hazardous Waste**. Includes edge-case rules for food-soiled paper, dirty packaging, and sharps. |
| 🌿 **ReWA AI Buddy** | AI recycling advisor powered by Google Gemini (with fallback to Groq, OpenRouter, and built-in Python heuristics) providing safe, practical preparation and reuse ideas. |
| 🗑️ **Six Digital Twin Bins** | Plastic, Paper, Metal, Glass, Hazardous, and Biodegradable bins tracking item counts and simulated capacity. |
| ✍️ **Human-in-the-Loop Verification** | Users can easily confirm or correct AI suggestions before adding items to bins. Corrections are logged to measure real-world AI accuracy and acceptance rates. |
| 🔒 **Per-Session Anonymous Isolation** | Anonymous visitors receive an isolated UUID session identity so items, points, and history never mix across visitors. |
| 📥 **History Export** | Download your session's recycling history as CSV or JSON with a single click. |
| 🌟 **Virtual Eco Rewards** | Earn 10 Virtual Eco Points per item sorted, daily streaks, 7 achievement badges, an evolving virtual eco garden, and an Eco Passport. |
| 📊 **Verifiable Analytics** | View verifiable metrics (items classified, items sorted, corrections logged, acceptance rate) alongside simulated capacities and material distribution charts. |
| 📈 **Model Performance Evaluation** | Built-in offline evaluation suite (`evaluate_model.py`) measuring top-1 accuracy, top-3 accuracy, per-label stats, and confusion pairs without faking data. |

---

## 🧠 How It Works

1. **Input Photo:** Upload an image file (JPG, PNG, WEBP), take a camera shot, or select a sample photo.
2. **AI Similarity Scoring:** Hugging Face `transformers` runs `openai/clip-vit-large-patch14` zero-shot classification against 22 candidate labels. The resulting number is a **similarity score**, not a statistical probability or guarantee. If the similarity score is below 55%, manual confirmation is prompted.
3. **Bengaluru Municipal Rules & Guidance:** Rules from `data/bengaluru_rules.json` determine whether the item goes to Wet Waste, Dry Waste, Sanitary, E-Waste, or Domestic Hazardous, highlighting edge cases (e.g. greasy pizza box bottoms go to wet waste).
4. **User Confirmation & Correction:** The user confirms or corrects both the detected item label and the target bin. Any disagreement is logged to calculate human correction rates.
5. **Virtual Bins & Rewards:** Confirmed items update the Digital Twin bin counts, increment simulated fill percentages, and award Virtual Eco Points.

---

## 🗂️ Project Structure

```
ReWA.AI/
├── README.md                          # Project documentation and user guide
├── requirements.txt                   # Production dependencies for Streamlit Cloud
├── app.py                             # Root launcher for Streamlit Cloud
├── test_rewa.py                       # Root test runner
├── evaluate_model.py                  # Standalone CLIP evaluation and benchmarking script
├── bin_rules.py                       # Single source of truth for bin mappings and labels
├── data/
│   └── bengaluru_rules.json           # Bengaluru municipal waste segregation rules & sources
├── sample_images/                     # Folder for one-click demo test photos
├── eval_data/                         # Folder for labeled evaluation datasets (eval_data/<label>/<img.jpg>)
├── docs/
│   └── screenshots/                   # Application screenshots
└── .vscode/
    ├── app1.py                        # Main Streamlit web application
    ├── auth.py                        # Supabase Auth (Email+Password) & Anonymous session isolation
    ├── bin_rules.py                   # Centralized bin rules and candidate labels
    ├── data_base.py                   # SQLite database operations & AI correction logging
    ├── digital_twin.py                # Digital Twin virtual bins and capacity simulation
    ├── recycling_logic.py             # Bin decision wrappers
    ├── rewabuddy_advisor.py           # LLM recycling & reuse assistant
    ├── rewards.py                     # Eco points, daily streaks, and daily goals
    ├── rewa_logic.py                  # Validation, badges, garden, and missions logic
    ├── test_rewa.py                   # Unittest test suite (37 tests)
    └── requirements.txt               # Dependencies
```

---

## 🚀 How to Run Locally

### 1. Prerequisites
- **Python 3.10+** (Tested on Python 3.10, 3.11, 3.12, 3.13)
- Internet connection (required on first run to download CLIP model weights)

### 2. Clone the Repository
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd ReWA.AI
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
You can run using either command:
```bash
streamlit run app.py
```
*Or directly targeting the core app file:*
```bash
streamlit run .vscode/app1.py
```

Open `http://localhost:8501` in your browser.

---

## 🧪 How to Run Tests

Run the complete automated test suite (checks session isolation, image validation, double-reward protection, badges, missions, and bin mapping completeness):

```bash
python test_rewa.py
```
*Or using Python's standard unittest discovery:*
```bash
python -m unittest discover -s .vscode -p "test_*.py" -v
```

---

## 📊 How to Run Model Evaluation

ReWA.AI includes a benchmark script (`evaluate_model.py`) to measure CLIP accuracy on real waste photos without fabricating results:

### 1. Build your Evaluation Dataset
Create folders inside `eval_data/` named after the candidate labels and add real photos (suggest ~5-10 photos per common label):

```
eval_data/
├── plastic bottle/
│   ├── bottle_01.jpg
│   └── bottle_02.png
├── cardboard box/
│   └── box_01.jpg
├── metal can/
│   └── can_01.jpg
└── battery/
    └── battery_01.jpg
```

*Candidate labels:* `cardboard box`, `paper sheet`, `newspaper`, `plastic bottle`, `plastic bag`, `plastic wrapper`, `metal can`, `glass bottle`, `glass jar`, `food waste`, `fruit peel`, `battery`, `light bulb`, `medicine`, `syringe or needle`, `medical waste`, `biohazard waste`, `chemical container`, `aerosol can`, `paint can`, `pesticide or poison`, `other waste`.

### 2. Run the Evaluator
```bash
python evaluate_model.py
```

*Outputs produced:*
- **Terminal Summary:** Overall top-1 and top-3 accuracy.
- `eval_report.json`: Machine-readable results (read automatically by the Streamlit app).
- `eval_report.md`: Markdown summary table with per-label accuracy and confusion pairs.
- `eval_results.csv`: Row-by-row prediction details for each tested photo.

*Note: If `eval_data/` is empty or missing, the script explains how to add data and exits gracefully with code 0.*

---

## ☁️ Free Deployment to Streamlit Community Cloud

ReWA.AI is 100% free and open-source, designed to deploy on the free tier of **Streamlit Community Cloud**:

1. Push your repository to GitHub.
2. Sign in to [share.streamlit.io](https://share.streamlit.io).
3. Click **"New app"**, select your repository and branch (`main`).
4. Set **Main file path** to `app.py`.
5. (Optional) Under **Advanced settings → Secrets**, paste your Supabase configuration:
   ```toml
   SUPABASE_URL = "https://your-project-id.supabase.co"
   SUPABASE_ANON_KEY = "eyJhbGciOi..."
   ```
   *(Note: The app is fully operational without secrets; anonymous mode is active by default with private per-session storage).*
6. Click **Deploy**.

---

## 🔐 Supabase Authentication Setup (100% Free Tier)

ReWA.AI uses Supabase Auth for optional email + password accounts:
- **Zero Cost:** Uses only Supabase Free Plan with the public `anon` key.
- **Privacy First:** If Supabase is unconfigured, every visitor receives an isolated per-session anonymous UUID. No data is mixed between visitors.
- **Auto-Sync:** When an anonymous visitor signs up or logs in, their accumulated recycling items, Eco Points, and streaks are automatically merged into their permanent account.

### How to configure:
1. Create a free account at [supabase.com](https://supabase.com) and click **"New Project"**.
2. Navigate to **Project Settings** (gear icon) → **API**.
3. Copy **Project URL** and the public **`anon` key**. *(Warning: NEVER copy or paste the `service_role` key).*
4. In your project, copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and paste the values.
5. *(Optional for demos)* In Supabase dashboard → **Authentication** → **Providers** → **Email**, turn off **"Confirm email"** to allow immediate logins without verification emails.

---

## ⚠️ Known Limitations & Honest Notes

- **Cold Starts & Streamlit Sleep Screen:** Streamlit Community Cloud puts free-tier apps into sleep mode after a period of inactivity. On wake-up, downloading the CLIP model weights (`~1.7 GB`) may take 1-2 minutes on the first classification attempt.
- **Similarity Scores vs Probability:** The confidence score displayed is a normalized cosine similarity score produced by CLIP over the fixed candidate label list. It is **not** a calibrated statistical probability or an absolute guarantee of correctness.
- **Ephemeral Storage on Cloud:** Streamlit Community Cloud instances do not have persistent SSD storage across container redeployments. SQLite database records may reset when the application restarts. Use the **"Download history (CSV/JSON)"** button to back up your recycling records.
- **Local Rules Variation:** While ReWA.AI incorporates official Bengaluru BBMP guidelines, rules and collection schedules can vary across individual wards, RWAs, and apartment complexes. Always check local notices.
- **Simulated Metrics:** Digital Twin bin capacity levels are virtual simulations (5% capacity per item, capped at 100%). No physical weight sensors or IoT bins are connected.
- **Virtual Eco Points:** Eco points are purely virtual and carry no real-world cash or monetary value.

---

## 👩‍💻 Authors
Made by **Jeevan DJ and J.Jebastin Immanuvel** — *Bangalore Institute of Technology (Information Science & Engineering)*

## 📄 License
This project is licensed under the MIT License.
