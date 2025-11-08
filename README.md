# ⚖️ Legal Document Analyzer

An AI-powered tool for analyzing legal documents, extracting key terms, risks, and obligations using advanced natural language processing.

## ✨ Features

- **AI-Powered Analysis**: Uses NVIDIA Nemotron Nano 12B V2 VL model via OpenRouter for comprehensive document analysis
- **Key Information Extraction**: Automatically identifies and extracts:
  - Risk factors and concerning clauses
  - Payment terms and conditions
  - Key obligations and responsibilities
  - Important legal clauses
- **Risk Assessment**: Provides an overall risk level assessment
- **User-Friendly Interface**: Clean, intuitive web interface built with Streamlit
- **Multiple Document Types**: Supports various legal documents including:
  - Contracts
  - NDAs
  - Agreements
  - Terms of Service
  - Privacy Policies

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip (Python package manager)
- OpenRouter API key (get one [here](https://openrouter.ai/keys))

### Installation

1. Clone this repository or download the files
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the project directory and add your OpenRouter API key:
   ```
   OPENROUTER_API_KEY=your_api_key_here
   ```

## Usage

1. Run the application:
   ```
   streamlit run app.py
   ```
2. Open the URL shown in the terminal (usually http://localhost:8501)
3. Upload a legal document (PDF)
4. View the AI-generated summary and analysis

## How It Works

The application:
1. Extracts text from the uploaded PDF using PyMuPDF
2. Sends the text to NVIDIA Nemotron Nano 12B V2 VL model via OpenRouter for analysis
3. Displays a structured summary including:
   - Executive Summary
   - Risk Factors
   - Payment Terms
   - Key Clauses
   - Overall Risk Assessment

## 🌐 Deployment

### Deploy to Streamlit Community Cloud (Free)

1. **Push to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository and branch
   - Set main file path: `app.py`
   - Click "Advanced settings" and add your secrets:
     ```toml
     OPENROUTER_API_KEY = "your_api_key_here"
     ```
   - Click "Deploy"

3. **Your app will be live** at: `https://your-app-name.streamlit.app`

### Alternative: Deploy with Docker

See `Dockerfile` (if provided) for containerized deployment options.

## ⚠️ Note

This tool is for informational purposes only and does not constitute legal advice. Always consult with a qualified legal professional for legal matters.
