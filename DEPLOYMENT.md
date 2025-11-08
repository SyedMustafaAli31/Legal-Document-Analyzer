# 🚀 Deployment Guide

This guide will help you deploy your Legal Document Analyzer to Streamlit Community Cloud.

## 📋 Prerequisites

Before deploying, make sure you have:
- ✅ A GitHub account
- ✅ Git installed on your computer
- ✅ An OpenRouter API key ([get one here](https://openrouter.ai/keys))

## 🌐 Option 1: Deploy to Streamlit Community Cloud (Recommended - FREE)

### Step 1: Prepare Your Repository

1. **Initialize Git** (if not already done):
   ```bash
   cd "c:\Users\SYED MUSTAFA\Downloads\legal_agreement"
   git init
   ```

2. **Create a GitHub repository**:
   - Go to [github.com](https://github.com)
   - Click the "+" icon → "New repository"
   - Name it (e.g., "legal-document-analyzer")
   - Don't initialize with README (we already have one)
   - Click "Create repository"

3. **Push your code to GitHub**:
   ```bash
   git add .
   git commit -m "Initial commit - Legal Document Analyzer"
   git remote add origin https://github.com/YOUR_USERNAME/legal-document-analyzer.git
   git branch -M main
   git push -u origin main
   ```

### Step 2: Deploy on Streamlit Cloud

1. **Go to Streamlit Cloud**:
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Click "Sign in with GitHub"
   - Authorize Streamlit to access your repositories

2. **Create New App**:
   - Click "New app" button
   - Select your repository: `YOUR_USERNAME/legal-document-analyzer`
   - Branch: `main`
   - Main file path: `app.py`

3. **Configure Secrets** (IMPORTANT):
   - Click "Advanced settings"
   - In the "Secrets" section, add your API key:
     ```toml
     OPENROUTER_API_KEY = "your-openrouter-api-key-here"
     ```
   - Replace `your-openrouter-api-key-here` with your actual OpenRouter API key
   - Click "Save"

4. **Deploy**:
   - Click "Deploy!"
   - Wait 2-5 minutes for deployment to complete

5. **Access Your App**:
   - Your app will be live at: `https://YOUR_APP_NAME.streamlit.app`
   - Share this URL with anyone!

### Step 3: Update Your App

Whenever you make changes:
```bash
git add .
git commit -m "Description of changes"
git push
```

Streamlit will automatically redeploy your app!

## 🐳 Option 2: Deploy with Docker

### Create Dockerfile

Create a `Dockerfile` in your project:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Build and Run

```bash
docker build -t legal-analyzer .
docker run -p 8501:8501 -e OPENROUTER_API_KEY=your_key_here legal-analyzer
```

### Deploy to Cloud Platforms

You can deploy the Docker container to:
- **Google Cloud Run**
- **AWS ECS/Fargate**
- **Azure Container Instances**
- **Railway.app**
- **Render.com**

## 🔒 Security Best Practices

1. **Never commit your API key** to GitHub
   - The `.gitignore` file is already configured to exclude `.env`
   
2. **Use Streamlit Secrets** for production
   - Store API keys in Streamlit Cloud's secrets manager
   
3. **Rotate your API keys** regularly
   - Update both in `.env` (local) and Streamlit secrets (production)

## 🆘 Troubleshooting

### App won't start
- Check that `requirements.txt` is present and correct
- Verify your API key is set in Streamlit secrets
- Check the app logs in Streamlit Cloud dashboard

### "Module not found" error
- Make sure all dependencies are listed in `requirements.txt`
- Redeploy the app

### API errors
- Verify your OpenRouter API key is valid
- Check if you have quota remaining
- Review the API key permissions

## 📊 Monitor Usage

- **Streamlit Cloud**: Check analytics in your Streamlit dashboard
- **OpenRouter**: Monitor API usage at [openrouter.ai](https://openrouter.ai)

## 🎉 Success!

Your Legal Document Analyzer is now live and accessible to anyone with the URL!

Share it with:
- Legal teams
- Colleagues
- Clients (with appropriate disclaimers)

---

**Need help?** Check:
- [Streamlit Deployment Docs](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app)
- [OpenRouter Documentation](https://openrouter.ai/docs)
