# ♻️ ReWA.AI — Recycling-Reusable Waste Assistant

**Sort Today • Save Tomorrow**

ReWA.AI helps you answer this simple question: **"What is the waste? How can I recycle/reuse it? which bin should the waste go?"**

You take a photo of a waste item. The app uses AI to guess what it is, suggests the right bin, and guides you how to recycle it, dispose of it and reuse it. Every item you sort earns virtual **Eco Points**, so recycling feels like a game.

> 📸
> [ReWA.AI Homepage](<img width="1887" height="912" alt="Screenshot 2026-10-16 152509" src="https://github.com/user-attachments/assets/eaa25895-1696-4426-98bd-b4d37f89bf75" />
)
> [AI-Powered Waste Assistant](<img width="1897" height="906" alt="Screenshot 2026-10-16 152556" src="https://github.com/user-attachments/assets/cb9c5e97-1bf3-4229-8caf-e5fc145e3697" />
)
> [Smart Recycling Guidance](<img width="1896" height="901" alt="Screenshot 2026-10-16 152614" src="https://github.com/user-attachments/assets/d3a63aa9-25f2-4e76-bbed-7e76e0a620fd" />
)

> [Digital Twin - Virtual Waste Bins](<img width="1912" height="896" alt="Screenshot 2026-10-16 152745" src="https://github.com/user-attachments/assets/b509d3e1-fd8f-4c74-bd86-e81863dfe97a" />
)
> [Analytics and Recycling Statistics](<img width="1897" height="891" alt="Screenshot 2026-10-16 152710" src="https://github.com/user-attachments/assets/cebbd70d-776d-47f9-bbb1-fb92a45debc1" />
)
> [Eco Points and Daily Streak](<img width="1910" height="900" alt="Screenshot 2026-10-16 152824" src="https://github.com/user-attachments/assets/ca0af40a-1098-4ab1-96dd-9af344b002d3" />
)
)

---

## ✨ What can it do?
| 📷 **Photo classification** | Upload photos or use your live camera. AI guesses the item (bottle, can, battery, food waste...). |
| 🌿 **ReWA AI Buddy** | An AI helper that explains the recycling advice and gives three easy reuse ideas in friendly language. |
| 🗑️ **Six virtual bins** | Plastic, Paper, Metal, Glass, Hazardous and Biodegradable. You always confirm the bin before adding an item. |
| ☣️ **Hazardous warnings** | Batteries, medicines, needles, chemicals and similar items are flagged and always go to the Hazardous bin, with extra-safe advice. |
| 🌟 **Eco Points** | 10 virtual points for every item you record. |
| 🔥 **Streaks and daily goal** | Recycle on consecutive days to build a streak. Daily goal: 5 items. |
| 🏅 **Badges, garden and passport** | 7 badges, a garden that grows as you recycle, and a passport with a stamp for each bin. |
| 📊 **Analytics** | Charts of items per material, bin capacity, points by category and the last 7 days. |
| 👤 **Sign-in (optional)** | Log in with Google or a phone number so each person keeps their own records. |

---

## 🧠 How does it work?

1. **You add a photo.** Upload files (JPG, PNG, WEBP) or take a picture with your camera. Up to 10 images at a time.
2. **The AI looks at it.** The app uses a model called **CLIP** (`openai/clip-vit-large-patch14`). It compares your photo with a list of about 22 waste labels and picks the best matches. The score it shows is a similarity score, **not** a guarantee that it is right.
3. **You check the bin.** The app suggests a bin. If the AI is not sure (score below 55%), it asks you to check manually.
4. **You add it to the bin.** The item is saved in the database and you earn Eco Points.
5. **Ask ReWA Buddy (optional).** The app sends the *checked facts* to an AI writer (Gemini, then Groq, then OpenRouter as backups). If all of them fail, it uses built-in Python rules instead, so you always get advice.

**Important design choice:** Python decides the material type, the bin and the safety rules. The AI writer only explains them in simple words. This stops the AI from inventing facts, especially for dangerous items.

---

## ⚠️ Honest notes

- **AI can be wrong.** Always check the real material and your local recycling rules.
- **Bin levels are simulated.** Each item adds 5% to a bin. There are no sensors or hardware.
- **Eco Points are virtual.** They have no cash value, and recording an item does not prove it was really recycled.

---

## 🗂️ Project files

| File | What it does |
|---|---|
| `app.py` | The main Streamlit app: pages, design, uploads, charts. Run this file. |
| `auth.py` | Sign-in with Google or phone number (optional). |
| `data_base.py` | Saves and reads recycling events in SQLite (`recycling.db`). |
| `rewards.py` | Eco Points, daily activity and streaks. Makes sure the same item is never rewarded twice. |
| `digital_twin.py` | The six virtual bins: counting items, simulated capacity, emptying. |
| `recycling_logic.py` | Decides which bin an item goes in and whether that bin is nearly full. |
| `rewa_logic.py` | Helper logic with no Streamlit: image checks, badges, garden, missions, passport. |
| `rewabuddy_advisor.py` | The ReWA AI Buddy: reuse ideas, safety tips and the Gemini → Groq → OpenRouter fallback. |
| `test_rewa_logic.py` | Automatic tests for `rewa_logic.py`. |

---

## 🚀 How to run it

### 1. Get the code
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Install the requirements
You need **Python 3.10+**.
```bash
pip install -r requirements.txt
```

### 3. (Optional) Add your AI keys
Without keys the app still works. ReWA Buddy just uses built-in rules.
```bash
export GEMINI_API_KEY="your-key"
export GROQ_API_KEY="your-key"
export OPENROUTER_API_KEY="your-key"
```
On Windows PowerShell use `$env:GEMINI_API_KEY="your-key"`.

### 4. Start the app
```bash
streamlit run app1.py
```
The first run downloads the CLIP model, so it can take a minute and needs internet.

---

## 🌍 Deploy it publicly (Streamlit Community Cloud)

ReWA.AI is a Streamlit app, so it is hosted on Streamlit Community Cloud (free), not Vercel.

1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click **Create app** → **Deploy a public app from GitHub**.
3. Fill in:
   - **Repository:** `jeevandj2008s-byte/rewa-ai`
   - **Branch:** `main`
   - **Main file path:** `app1.py`
   - **App URL:** pick a name, e.g. `rewa-ai`
4. Open **Advanced settings**:
   - **Python version:** 3.11
   - **Secrets:** paste the contents of your local `.streamlit/secrets.toml`, changing
     `redirect_uri` to `https://<your-app-name>.streamlit.app/oauth2callback`.
     Add AI keys here too if you use them, e.g. `GEMINI_API_KEY = "your-key"`.
5. Click **Deploy**. The first build installs torch and downloads CLIP, so it takes a few minutes.
6. In Google Cloud Console, add the same `https://<your-app-name>.streamlit.app/oauth2callback`
   to your OAuth client's **Authorized redirect URIs**, or Google sign-in will fail.

The app is public by default — anyone with the link can open it. Every `git push` to `main` redeploys it automatically.

> Note: the SQLite database lives on the server's disk, which Streamlit Cloud resets on restart/redeploy, so saved points and history can be lost. Use a hosted database if you need them to persist.

---

## 🔐 Setting up sign-in 

Create `.streamlit/secrets.toml`:
```toml
[auth]
redirect_uri = "http://localhost:8501/oauth2callback"
cookie_secret = "any-long-random-string"
client_id = "xxxx.apps.googleusercontent.com"
client_secret = "xxxx"
server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"

# Optional phone login (SMS code through Twilio Verify)
[twilio]
account_sid = "ACxxxx"
auth_token = "xxxx"
verify_sid = "VAxxxx"
```
- Google: create an OAuth Web client in Google Cloud Console and add the same `redirect_uri` there.
- Each user's items, points and streaks are stored separately by `user_id`.
- **Never upload `secrets.toml` to GitHub.**

---

## 🧪 Run the tests
```bash
python -m unittest -v test_rewa_logic
```
The tests check reward logic (no double points), image validation, badges, garden stages and missions.

---

## 🛠️ Built with

- **Python** and **Streamlit** (web app)
- **Hugging Face Transformers** with **CLIP** (image understanding)
- **Pillow** (image handling), **pandas** and **Altair** (charts)
- **SQLite** (database)
- **Google Gemini**, **Groq** and **OpenRouter** (advice writing)

---

## 🔮 Ideas for the future
- Real bin sensors instead of simulated capacity
- Local recycling rules by city
- Team or school leaderboards
- A cloud database (for example Postgres) for many users

---

## 👩‍💻 Author
Made by **Jeevan DJ and J.Jebastin Immanuvel** — *Bangalore Institution of Technology-Information Science Engineering(1st Year)*

## 📄 License
This project is licensed under the MIT License.
