# 🚀 Quick Deployment Guide

Your Legal Document Analyzer is ready to deploy! Follow these simple steps:

## ✅ Pre-Deployment Checklist

- [x] App is working locally
- [x] API key is configured
- [x] All files are ready

## 🌐 Deploy to Streamlit Cloud (FREE & EASIEST)

### Step 1: Install Git (if not installed)
Download and install Git from: https://git-scm.com/download/win

### Step 2: Create GitHub Account
Go to https://github.com and create a free account (if you don't have one)

### Step 3: Push Your Code to GitHub

Open **Git Bash** (or PowerShell after installing Git) in your project folder:

```bash
cd "c:\Users\SYED MUSTAFA\Downloads\legal_agreement"

# Initialize git repository
git init

# Add all files
git add .

# Commit your code
git commit -m "Initial commit - Legal Document Analyzer"

# Create a new repository on GitHub (via web browser):
# 1. Go to https://github.com/new
# 2. Name it: legal-document-analyzer
# 3. Don't initialize with README
# 4. Click "Create repository"

# Link to your GitHub repo (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/legal-document-analyzer.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### Step 4: Deploy on Streamlit Cloud

1. **Go to**: https://share.streamlit.io
2. **Sign in** with your GitHub account
3. Click **"New app"**
4. Fill in the details:
   - **Repository**: YOUR_USERNAME/legal-document-analyzer
   - **Branch**: main
   - **Main file path**: app.py
5. Click **"Advanced settings"**
6. In the **Secrets** section, paste your API key:
   ```toml
   OPENROUTER_API_KEY = "your-openrouter-api-key-here"
   ```
   Replace `your-openrouter-api-key-here` with your actual OpenRouter API key
7. Click **"Deploy!"**

### Step 5: Wait & Access

⏱️ Wait 2-5 minutes for deployment to complete.

🎉 Your app will be live at: `https://your-app-name.streamlit.app`

---

## 🔄 Update Your Deployed App

Whenever you make changes:

```bash
git add .
git commit -m "Description of changes"
git push
```

Streamlit will automatically redeploy! ✨

---

## 🆘 Troubleshooting

### Error: "No module named 'X'"
- Check that `requirements.txt` includes all dependencies
- Redeploy the app

### Error: "API key not found"
- Double-check the secrets in Streamlit Cloud settings
- Make sure the format is correct (TOML format)

### App won't start
- Check the logs in Streamlit Cloud dashboard
- Verify all files are pushed to GitHub

---

## 📱 Alternative: Deploy WITHOUT Git

If you don't want to use Git, you can use **Streamlit Cloud via GitHub Upload**:

1. Go to GitHub.com
2. Click "+" → "New repository"
3. Name it "legal-document-analyzer"
4. After creation, click "uploading an existing file"
5. Drag and drop all your files (except `.env`)
6. Commit the files
7. Follow Step 4 above to deploy on Streamlit Cloud

---

## 🎯 What's Next?

After deployment:
- ✅ Test your app with different documents
- ✅ Share the URL with others
- ✅ Monitor usage in Streamlit dashboard
- ✅ Check API usage on OpenRouter

**Need help?** Check the full DEPLOYMENT.md guide!
