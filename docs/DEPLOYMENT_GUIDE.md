# Complete Deployment Guide & Recommendations
**House Price Least-Squares Model Comparison System**

---

## 🌟 Recommended Deployment Strategy: The Hybrid Cloud Approach

For an academic linear algebra project combining mathematical rigor with interactive visualization, we recommend **Option A (Streamlit Community Cloud)** paired with **Option B (GitHub Pages)**:

| Platform | Best For | Cost | Setup Time | URL Example |
|---|---|:---:|:---:|---|
| **Streamlit Community Cloud** (Top Pick) | Live Python app, interactive feature sliders, dynamic plots | Free | 2 mins | `https://house-price-least-squares.streamlit.app` |
| **GitHub Pages** | Static documentation, offline-ready web calculator, zero server | Free | 1 min | `https://<username>.github.io/<repo>/` |
| **Render / Railway / Docker** | Full server deployment with custom domain | Free tier | 5 mins | `https://house-price.onrender.com` |

---

## Step 1: Initialize Git Repository & Push to GitHub

Open your terminal in the project directory:

```bash
cd /Users/farhanullakhan/.gemini/antigravity/scratch/house_price_least_squares

# 1. Initialize git
git init -b main

# 2. Add all project files
git add .

# 3. Create initial commit
git commit -m "feat: complete house price least-squares system with 8-stage pipeline"

# 4. Create a new repository on GitHub (named e.g. house-price-least-squares)
# Link remote and push (replace YOUR_GITHUB_USERNAME with your username)
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/house-price-least-squares.git
git push -u origin main
```

---

## Step 2: Option A — Deploy to Streamlit Community Cloud (Recommended ⭐)

Streamlit Community Cloud provides 1-click cloud hosting directly from your GitHub repository for free:

1. Go to **[share.streamlit.io](https://share.streamlit.io)** and log in with your GitHub account.
2. Click **"New app"**.
3. Select your repository: `YOUR_GITHUB_USERNAME/house-price-least-squares`.
4. Set **Branch**: `main`.
5. Set **Main file path**: `app.py`.
6. Click **"Deploy!"**.

Within 60 seconds, your interactive web dashboard will be live at:
`https://<your-app-name>.streamlit.app`

Any future commits pushed to GitHub will automatically trigger live re-deployment!

---

## Step 3: Option B — Enable GitHub Pages for Free Static Hosting

GitHub Pages will host the interactive HTML dashboard (`docs/index.html`) at zero cost:

1. Navigate to your repository on GitHub.
2. Go to **Settings** > **Pages** (under "Code and automation" in the left sidebar).
3. Under **Build and deployment**:
   - **Source**: `Deploy from a branch`
   - **Branch**: `main`
   - **Folder**: `/docs`
4. Click **Save**.
5. Your live site will be ready in 1 minute at:
   `https://YOUR_GITHUB_USERNAME.github.io/house-price-least-squares/`

---

## Step 4: Option C — Deploy via Docker or Render

### Docker Container:
Build and run the Docker image locally or on any cloud VPS (AWS, GCP, DigitalOcean):
```bash
docker build -t house-price-app .
docker run -p 8501:8501 house-price-app
```
Access at `http://localhost:8501`.

### Render Free Web Service:
1. Sign up on **[render.com](https://render.com)**.
2. Create a new **Web Service** connected to your GitHub repository.
3. Select **Docker** or **Python 3**.
4. Set start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`.
