import os
import re
import fitz  # PyMuPDF
import streamlit as st
from dotenv import load_dotenv
from typing import Dict, List, Optional
from pydantic import BaseModel
from openai import OpenAI

# Load environment variables
load_dotenv()

# Configure OpenRouter with DeepSeek
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')

# Try Streamlit secrets if .env not found
if not OPENROUTER_API_KEY:
    try:
        OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY")
    except:
        pass

client = None
MODEL_NAME = "nvidia/nemotron-nano-12b-v2-vl:free"

# Initialize session state for model name
if 'model_name' not in st.session_state:
    st.session_state.model_name = MODEL_NAME

if OPENROUTER_API_KEY:
    try:
        # Strip whitespace and validate key format
        OPENROUTER_API_KEY = OPENROUTER_API_KEY.strip()
        if not OPENROUTER_API_KEY.startswith('sk-or-v1-'):
            st.error("⚠️ Invalid OpenRouter API key format. Key should start with 'sk-or-v1-'")
        else:
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=OPENROUTER_API_KEY,
                default_headers={
                    "HTTP-Referer": "https://legal-doc-analyzer.streamlit.app",
                    "X-Title": "Legal Document Analyzer",
                }
            )
            st.session_state.model_name = MODEL_NAME
    except Exception as e:
        st.error(f"Error initializing OpenRouter client: {str(e)}")
else:
    st.warning("OPENROUTER_API_KEY not found. Please add it to your .env file or Streamlit secrets.")

# Pydantic models for structured output
class LegalDocumentAnalysis(BaseModel):
    summary: str
    risk_factors: List[str]
    payment_terms: List[str]
    obligations: List[str]
    key_clauses: Dict[str, str]
    overall_risk_level: str  # Low/Medium/High

def analyze_with_nvidia(text: str, doc_type: str = "agreement") -> Optional[LegalDocumentAnalysis]:
    """Analyze legal document using NVIDIA model via OpenRouter"""
    if not client:
        st.error("OpenRouter client not initialized. Please check your API key.")
        return None

    try:
        # Truncate text to fit within context window
        max_length = 5000  # Further reduced limit to be safe
        truncated_text = text[:max_length]
        
        # Create a simpler, more direct prompt
        prompt = f"""Analyze this {doc_type} document and provide the following information in markdown format:
        
# Document Analysis

## Summary
A brief summary of the document's purpose and key points

## Risk Factors
List any potential risks or concerning clauses

## Payment Terms
Any payment schedules or financial conditions

## Key Clauses
Important legal clauses with brief explanations

## Overall Risk Level
[Low/Medium/High] based on the document's terms

DOCUMENT:
{truncated_text}"""
        
        # Configure generation settings
        generation_config = {
            "temperature": 0.2,  # Lower temperature for more focused output
            "max_output_tokens": 2000,
        }
        
        # Generate content with the model
        try:
            completion = client.chat.completions.create(
                model=st.session_state.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=2000,
            )

            # Debug: Store the raw response
            st.session_state.last_response = completion

            response_text = completion.choices[0].message.content

            if not response_text.strip():
                st.error("Received empty response from the API")
                return None
                
            # Store the response for debugging
            st.session_state.last_response_text = response_text
            
            # If the response is in markdown format, return it directly
            if any(header in response_text.lower() for header in ['## summary', '## risk factors', '## payment terms']):
                return response_text
                
            # Otherwise, try to parse it
            return parse_llm_response(response_text)
            
        except Exception as gen_error:
            st.error(f"Error generating content: {str(gen_error)}")
            return None

    except Exception as e:
        st.error(f"Unexpected error in analyze_with_nvidia: {str(e)}")
        return None

def parse_llm_response(response: str) -> LegalDocumentAnalysis:
    """Parse the LLM response into structured format"""
    # Initialize with default values
    analysis = {
        "summary": "",
        "risk_factors": [],
        "payment_terms": [],
        "obligations": [],
        "key_clauses": {},
        "overall_risk_level": "Medium"
    }
    
    # Split into sections based on all caps headers
    sections = {}
    current_section = None
    
    for line in response.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        # Check for section headers (all caps followed by colon)
        if ':' in line and line.split(':')[0].isupper() and ' ' not in line.split(':')[0]:
            current_section = line.split(':')[0].strip().lower()
            sections[current_section] = line.split(':', 1)[1].strip()
        elif current_section:
            sections[current_section] += '\n' + line
    
    # Map sections to analysis fields
    if 'summary' in sections:
        analysis['summary'] = sections['summary']
    
    if 'risk factors' in sections:
        analysis['risk_factors'] = [f.strip() for f in sections['risk factors'].split('\n') if f.strip()]
    
    if 'payment terms' in sections:
        analysis['payment_terms'] = [pt.strip() for pt in sections['payment terms'].split('\n') if pt.strip()]
    
    if 'obligations' in sections:
        analysis['obligations'] = [ob.strip() for ob in sections['obligations'].split('\n') if ob.strip()]
    
    if 'key clauses' in sections:
        clauses = {}
        for line in sections['key clauses'].split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                clauses[key.strip()] = value.strip()
        analysis['key_clauses'] = clauses
    
    if 'overall risk level' in sections:
        risk_level = sections['overall risk level'].lower()
        if 'high' in risk_level:
            analysis['overall_risk_level'] = 'High'
        elif 'low' in risk_level:
            analysis['overall_risk_level'] = 'Low'
    
    # Fallback parsing if section-based parsing didn't work
    if not any(analysis.values()):
        current_section = None
        for line in response.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            if line.upper().startswith('SUMMARY:'):
                current_section = 'summary'
                analysis['summary'] = line.split(':', 1)[1].strip()
            elif line.upper().startswith('RISK FACTORS:'):
                current_section = 'risk_factors'
            elif line.upper().startswith('PAYMENT TERMS:'):
                current_section = 'payment_terms'
            elif line.upper().startswith('OBLIGATIONS:'):
                current_section = 'obligations'
            elif line.upper().startswith('KEY CLAUSES:'):
                current_section = 'key_clauses'
            elif line.upper().startswith('OVERALL RISK LEVEL:'):
                risk_level = line.split(':', 1)[1].strip().lower()
                if 'high' in risk_level:
                    analysis['overall_risk_level'] = 'High'
                elif 'low' in risk_level:
                    analysis['overall_risk_level'] = 'Low'
                current_section = None
            elif current_section == 'risk_factors' and line.startswith('-'):
                analysis['risk_factors'].append(line[1:].strip())
            elif current_section == 'payment_terms' and line.startswith('-'):
                analysis['payment_terms'].append(line[1:].strip())
            elif current_section == 'obligations' and line.startswith('-'):
                analysis['obligations'].append(line[1:].strip())
            elif current_section == 'key_clauses' and ':' in line:
                key, value = line.split(':', 1)
                analysis['key_clauses'][key.strip()] = value.strip()
    
    return LegalDocumentAnalysis(**analysis)

    # Enhanced word frequency analysis
    words = re.findall(r'\b\w+\b', text.lower())
    word_freq = Counter(words)
    
    # Remove common words and short words
    stop_words = set([
        'the', 'and', 'of', 'to', 'in', 'a', 'is', 'that', 'it', 'as', 'be', 'are', 
        'for', 'on', 'this', 'by', 'with', 'or', 'an', 'at', 'from', 'which', 'was', 
        'were', 'has', 'have', 'had', 'been', 'will', 'shall', 'may', 'can', 'could',
        'would', 'should', 'must', 'such', 'any', 'all', 'its', 'their', 'other'
    ])
    
    # Legal-specific stop words to remove
    legal_stop_words = set([
        'herein', 'hereby', 'hereto', 'hereof', 'hereunder', 'therein', 'thereby',
        'thereto', 'thereof', 'whereas', 'witnesseth', 'notwithstanding', 'pursuant'
    ])
    
    # Remove stop words and count frequencies of remaining words
    for word in list(word_freq):
        if (word in stop_words or 
            word in legal_stop_words or 
            len(word) < 3 or 
            word.isdigit()):
            del word_freq[word]
    
    # Score sentences based on word frequency and position
    sentence_scores = {}
    for i, sentence in enumerate(sentences):
        # Score based on word frequency
        words_in_sentence = re.findall(r'\b\w+\b', sentence.lower())
        if not words_in_sentence:
            continue
            
        # Calculate score based on word frequency
        freq_score = sum(word_freq.get(word, 0) for word in words_in_sentence)
        freq_score = freq_score / len(words_in_sentence)  # Normalize by sentence length
        
        # Give higher weight to sentences with legal terms
        legal_terms = ['shall', 'must', 'will', 'agree', 'party', 'obligation', 
                      'right', 'duty', 'liability', 'indemnification', 'breach']
        legal_score = sum(1 for term in legal_terms if term in sentence.lower())
        
        # Combine scores
        sentence_scores[i] = freq_score + (legal_score * 0.5)
    
    # Get top sentences, ensuring we don't take too many from the same area
    top_sentences = []
    sorted_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Take top sentences with some distribution
    for idx, score in sorted_sentences:
        # Don't take too many sentences from the same part of the document
        if not any(abs(idx - s[0]) < 5 for s in top_sentences):
            top_sentences.append((idx, score))
            if len(top_sentences) >= max_sentences:
                break
    
    # Sort by original position
    top_sentences.sort()
    
    # Return the summary with some basic formatting
    summary = []
    for i, (idx, score) in enumerate(top_sentences):
        summary.append(sentences[idx])
        
    return '\n\n'.join(summary)

def extract_legal_sections(text):
    """Extract common legal sections from the document"""
    # Common legal section headers and their variations
    section_patterns = {
        'Parties': r'(?:parties|between|agreement between|between and among)(.*?)(?=\n\n|\n[A-Z][^\n]{10,}\n|$)',
        'Definitions': r'(?:definitions|defined terms)(.*?)(?=\n\n|\n[A-Z][^\n]{10,}\n|$)',
        'Term': r'(?:term|duration|period of agreement)(.*?)(?=\n\n|\n[A-Z][^\n]{10,}\n|$)',
        'Termination': r'(?:termination|expiration)(.*?)(?=\n\n|\n[A-Z][^\n]{10,}\n|$)',
        'Confidentiality': r'(?:confidentiality|non-disclosure|nda)(.*?)(?=\n\n|\n[A-Z][^\n]{10,}\n|$)',
        'Payment Terms': r'(?:payment|fee|compensation|consideration)(.*?)(?=\n\n|\n[A-Z][^\n]{10,}\n|$)',
        'Governing Law': r'(?:governing law|jurisdiction|dispute resolution)(.*?)(?=\n\n|\n[A-Z][^\n]{10,}\n|$)',
        'Representations and Warranties': r'(?:representations|warranties)(.*?)(?=\n\n|\n[A-Z][^\n]{10,}\n|$)'
    }
    
    extracted = {}
    for section, pattern in section_patterns.items():
        matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
        if matches:
            # Clean up the extracted text
            clean_text = ' '.join([m.strip() for m in matches if m.strip()])
            if len(clean_text) > 20:  # Only include if we have meaningful content
                extracted[section] = clean_text[:500] + ('...' if len(clean_text) > 500 else '')
    
    return extracted

def generate_legal_summary(text):
    """Generate a structured legal summary using local processing"""
    summary = "# Document Analysis\n\n"
    
    # Extract structured sections
    sections = extract_legal_sections(text)
    
    if not sections:
        # If no sections found, try to identify key paragraphs
        summary += "## Document Overview\n"
        
        # Look for common legal phrases
        key_phrases = [
            'agreement', 'obligation', 'right', 'duty', 'liability',
            'indemnification', 'breach', 'remedy', 'amendment', 'assignment'
        ]
        
        found_sections = {}
        for phrase in key_phrases:
            # Find sentences containing these phrases
            pattern = r'([^.!?]*' + re.escape(phrase) + r'[^.!?]*[.!?])'
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                found_sections[phrase.capitalize()] = ' '.join(matches[:3])  # Take up to 3 matching sentences
        
        if found_sections:
            for section, content in found_sections.items():
                summary += f"### {section}\n{content}\n\n"
        else:
            # Fall back to general summarization
            summary += "## Key Points\n" + summarize_text_locally(text)
    else:
        # Format the extracted sections
        for section, content in sections.items():
            summary += f"## {section}\n{content}\n\n"
    
    # Add document statistics
    word_count = len(re.findall(r'\b\w+\b', text))
    sentences = re.split(r'(?<![A-Z][a-z]\\.)(?<=[.!?]) +', text)
    avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
    
    summary += "## Document Statistics\n"
    summary += f"- Total words: {word_count}\n"
    summary += f"- Sentences: {len(sentences)}\n"
    summary += f"- Average sentence length: {avg_sentence_length:.1f} words\n\n"
    
    return summary

def detect_document_type(text, filename):
    """Detect document type by analyzing both filename and content."""
    # Convert to lowercase for case-insensitive matching
    content = text.lower()
    filename = filename.lower()
    
    # Check for NDA (Non-Disclosure Agreement)
    nda_terms = ['nda', 'non-disclosure', 'confidentiality agreement']
    if any(term in filename for term in nda_terms) or \
       any(term in content[:2000] for term in nda_terms):
        return 'NDA'
    
    # Check for Terms of Service
    tos_terms = ['terms of service', 'terms and conditions', 'terms & conditions', 'tos', 'terms of use']
    if any(term in filename for term in tos_terms) or \
       any(term in content[:2000] for term in tos_terms):
        return 'Terms of Service'
    
    # Check for Privacy Policy
    privacy_terms = ['privacy policy', 'gdpr', 'data protection', 'privacy notice']
    if any(term in filename for term in privacy_terms) or \
       any(term in content[:2000] for term in privacy_terms):
        return 'Privacy Policy'
    
    # Check for Contract
    contract_terms = ['contract', 'agreement between', 'this agreement', 'party a and party b']
    if any(term in filename for term in contract_terms) or \
       any(term in content[:2000] for term in contract_terms):
        return 'Contract'
    
    # Default to Agreement if no specific type is detected
    return 'Agreement'

def extract_text_from_pdf(uploaded_file):
    """Extract text from uploaded PDF file with better error handling and progress updates."""
    temp_file = "temp.pdf"
    try:
        # Show file info
        file_size = len(uploaded_file.getvalue()) / (1024 * 1024)  # Size in MB
        st.info(f"Processing PDF ({file_size:.2f} MB)...")
        
        # Save the uploaded file to a temporary file
        with open(temp_file, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Open the saved file
        try:
            doc = fitz.open(temp_file)
            total_pages = len(doc)
            
            # Check if document is encrypted
            if doc.is_encrypted:
                # Try to decrypt with empty password
                if not doc.authenticate(""):
                    st.error("This PDF is password protected. Please provide the password or use an unprotected PDF.")
                    return None
            
            # Create a progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Process each page
            text = ""
            max_pages = min(total_pages, 50)  # Limit to first 50 pages
            
            for i in range(max_pages):
                # Update progress
                progress = (i + 1) / max_pages
                progress_bar.progress(progress)
                status_text.text(f"Processing page {i+1} of {max_pages}...")
                
                # Get page text
                page = doc.load_page(i)
                page_text = page.get_text("text")
                
                # Clean up the text
                if page_text.strip():
                    text += page_text + "\n\n"
            
            # Close the document
            doc.close()
            
            # Clean up progress bar
            progress_bar.empty()
            status_text.empty()
            
            if not text.strip():
                st.warning("The PDF appears to be empty or contains no extractable text. It might be a scanned document.")
                return None
                
            return text
            
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            return None
            
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None
        
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except Exception as e:
                st.warning(f"Warning: Could not remove temporary file: {str(e)}")


def display_analysis(analysis):
    """Display the analysis results in a user-friendly format
    
    Args:
        analysis: Can be a LegalDocumentAnalysis object or a markdown string
    """
    # If analysis is a string, parse and display it nicely
    if isinstance(analysis, str):
        # Parse the markdown response to extract sections
        sections = {}
        current_section = None
        current_content = []
        
        for line in analysis.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            # Check for main section headers (## Header)
            if line.startswith('## '):
                # Save previous section
                if current_section:
                    sections[current_section] = '\n'.join(current_content)
                current_section = line[3:].strip()
                current_content = []
            elif current_section:
                current_content.append(line)
        
        # Save last section
        if current_section:
            sections[current_section] = '\n'.join(current_content)
        
        # Display sections in a structured way
        if 'Summary' in sections:
            st.markdown("### 📝 Document Summary")
            # Clean up the summary text
            summary_text = sections['Summary'].replace('**', '').strip()
            for item in summary_text.split('\n'):
                if item.strip():
                    # Remove numbered list markers
                    clean_item = re.sub(r'^\d+\.\s*', '', item.strip())
                    if clean_item:
                        st.markdown(f"- {clean_item}")
            st.markdown("---")
        
        # Display Risk Factors
        if 'Risk Factors' in sections:
            st.markdown("### 🚨 Risk Factors")
            risk_text = sections['Risk Factors']
            for item in risk_text.split('\n'):
                if item.strip() and item.strip().startswith('-'):
                    clean_item = item.strip()[1:].strip()
                    clean_item = clean_item.replace('**', '')
                    if clean_item:
                        st.markdown(f"⚠️ **{clean_item.split(':')[0]}:** {':'.join(clean_item.split(':')[1:]).strip() if ':' in clean_item else ''}")
            st.markdown("---")
        
        # Display Payment Terms
        if 'Payment Terms' in sections:
            st.markdown("### 💰 Payment Terms")
            payment_text = sections['Payment Terms']
            for item in payment_text.split('\n'):
                if item.strip() and item.strip().startswith('-'):
                    clean_item = item.strip()[1:].strip()
                    clean_item = clean_item.replace('**', '')
                    if clean_item:
                        st.markdown(f"- {clean_item}")
            st.markdown("---")
        
        # Display Key Clauses
        if 'Key Clauses' in sections:
            st.markdown("### 📋 Key Clauses")
            clauses_text = sections['Key Clauses']
            items = clauses_text.split('\n')
            for i, item in enumerate(items):
                if item.strip():
                    clean_item = item.strip()
                    # Remove list markers
                    clean_item = re.sub(r'^[-\d+\.]\s*', '', clean_item)
                    clean_item = clean_item.replace('**', '')
                    if clean_item:
                        # Check if it contains a colon (title: description)
                        if ':' in clean_item:
                            title = clean_item.split(':')[0].strip()
                            desc = ':'.join(clean_item.split(':')[1:]).strip()
                            with st.expander(f"🔹 {title}"):
                                st.write(desc)
                        else:
                            st.markdown(f"- {clean_item}")
            st.markdown("---")
        
        # Display Overall Risk Level if found
        if 'Overall Risk Level' in sections:
            st.markdown("### ⚠️ Overall Risk Level")
            risk_level_text = sections['Overall Risk Level'].strip()
            
            # Extract risk level
            if 'High' in risk_level_text or 'high' in risk_level_text:
                risk_level = 'High'
                color = 'red'
            elif 'Low' in risk_level_text or 'low' in risk_level_text:
                risk_level = 'Low'
                color = 'green'
            else:
                risk_level = 'Medium'
                color = 'orange'
            
            st.markdown(f"<h2 style='color: {color}'>{risk_level}</h2>", unsafe_allow_html=True)
            
            # Display explanation if any
            explanation = re.sub(r'\[.*?\]', '', risk_level_text).strip()
            if explanation and len(explanation) > 10:
                st.caption(explanation)
        
        return
        
    # If it's a dictionary, try to convert it to a LegalDocumentAnalysis object
    if isinstance(analysis, dict):
        try:
            analysis = LegalDocumentAnalysis(**analysis)
        except Exception as e:
            st.error(f"Error converting analysis to LegalDocumentAnalysis: {str(e)}")
            st.json(analysis)  # Show the raw analysis for debugging
            return
    
    # If it's still not a LegalDocumentAnalysis object, show an error
    if not isinstance(analysis, LegalDocumentAnalysis):
        st.error(f"Unexpected analysis type: {type(analysis).__name__}")
        st.json(str(analysis)[:1000])  # Show first 1000 chars for debugging
        return
    
    # Display risk level with color coding
    risk_colors = {
        "High": "red",
        "Medium": "orange",
        "Low": "green"
    }
    
    # Ensure all required attributes exist
    if not hasattr(analysis, 'summary') or not analysis.summary:
        analysis.summary = "No summary available."
    if not hasattr(analysis, 'overall_risk_level') or not analysis.overall_risk_level:
        analysis.overall_risk_level = "Medium"
    if not hasattr(analysis, 'key_clauses') or not analysis.key_clauses:
        analysis.key_clauses = {"No clauses found": "No key clauses were identified in the document."}
    if not hasattr(analysis, 'obligations') or not analysis.obligations:
        analysis.obligations = ["No specific obligations identified."]
    if not hasattr(analysis, 'payment_terms') or not analysis.payment_terms:
        analysis.payment_terms = ["No specific payment terms identified."]
    if not hasattr(analysis, 'risk_factors') or not analysis.risk_factors:
        analysis.risk_factors = ["No specific risk factors identified."]
    
    # Display the analysis
    st.markdown("### 📝 Document Summary")
    st.markdown(analysis.summary)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ⚠️ Overall Risk Level")
        st.markdown(f"<h1 style='color: {risk_colors.get(analysis.overall_risk_level, 'gray')}'>"
                   f"{analysis.overall_risk_level}</h1>", 
                   unsafe_allow_html=True)
        
        st.markdown("### 📋 Key Clauses")
        for clause, desc in analysis.key_clauses.items():
            with st.expander(f"🔹 {clause}"):
                st.write(desc)
    
    with col2:
        st.markdown("### ⚖️ Key Obligations")
        for obligation in analysis.obligations:
            st.markdown(f"- {obligation}")
            
        st.markdown("\n### 💰 Payment Terms")
        for term in analysis.payment_terms:
            st.markdown(f"- {term}")
    
    st.markdown("---")
    st.markdown("### 🚨 Potential Risk Factors")
    for risk in analysis.risk_factors:
        st.markdown(f"- ⚠️ {risk}")

def chat_about_document(question: str, document_text: str, doc_type: str = "document") -> str:
    """Generate a response to a question about the document"""
    if not client:
        return "Error: OpenRouter client not initialized. Please check your API key."
    
    try:
        # Truncate document text to stay within token limits
        max_length = min(10000, len(document_text))
        truncated_text = document_text[:max_length]
        
        # Create the prompt text
        prompt_text = f"""You are a helpful legal assistant. Answer the question based on the provided document.
        
DOCUMENT TYPE: {doc_type}

DOCUMENT CONTENT:
{truncated_text}

QUESTION: {question}

Please provide a clear, concise answer based on the document. If the answer isn't in the document, say so.

ANSWER:"""
        
        # Create the generation config
        generation_config = {
            "temperature": 0.3,
            "max_output_tokens": 1000,
        }
        
        # Make the API request
        completion = client.chat.completions.create(
            model=st.session_state.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful legal assistant."},
                {"role": "user", "content": prompt_text}
            ],
            temperature=0.3,
            max_tokens=1000,
        )

        # Extract text from response
        try:
            return completion.choices[0].message.content
            
        except Exception as extract_error:
            return f"Error processing the response: {str(extract_error)}"
            
    except Exception as e:
        return f"Error communicating with the AI model: {str(e)}"

def main():
    st.set_page_config(
        page_title="⚖️ Legal Document Analyzer",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize chat history if it doesn't exist
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Initialize document text if it doesn't exist
    if 'document_text' not in st.session_state:
        st.session_state.document_text = ""
    
    # Sidebar for settings
    with st.sidebar:
        st.title("⚙️ Settings")
        
        # Initialize document type in session state if not exists
        if 'detected_doc_type' not in st.session_state:
            st.session_state.detected_doc_type = "Agreement"
        
        # Get the detected document type from session state
        doc_type = st.selectbox(
            "Document Type",
            ["Agreement", "NDA", "Contract", "Terms of Service", "Privacy Policy"],
            index=["Agreement", "NDA", "Contract", "Terms of Service", "Privacy Policy"].index(
                st.session_state.detected_doc_type
            ) if st.session_state.detected_doc_type in ["Agreement", "NDA", "Contract", "Terms of Service", "Privacy Policy"] else 0
        )
        
        # Show detected type message if a document is being processed
        if 'document_text' in st.session_state and st.session_state.document_text:
            st.caption(f"Auto-detected as: {st.session_state.detected_doc_type}")
            st.caption("(You can change this if needed)")
        
        st.markdown("---")
        st.markdown("### 🔑 API Setup")
        st.markdown("1. Get an [OpenRouter API key](https://openrouter.ai/keys)")
        st.markdown("2. Create a `.env` file with:")
        st.code("OPENROUTER_API_KEY=your_api_key_here")
    
    # Main content
    st.title("⚖️ Legal Document Analyzer")
    st.markdown("Upload a legal document to get an AI-powered analysis of its key terms, risks, and obligations.")
    
    # File upload section
    uploaded_file = st.file_uploader(
        "📄 Upload a legal document (PDF)", 
        type=["pdf"],
        help="Supported formats: PDF"
    )
    
    # Process document when uploaded
    if uploaded_file is not None:
        with st.spinner("🔍 Analyzing document..."):
            # Extract text from PDF
            text = extract_text_from_pdf(uploaded_file)
            
            if text:
                st.success("✅ Document processed successfully!")
                
                # Show extracted text in expander
                with st.expander("📄 View extracted text"):
                    st.text_area("Extracted Text", text, height=300, label_visibility="collapsed")
                
                # Store the extracted text in session state
                st.session_state.document_text = text
                
                # Auto-detect document type from content
                with st.spinner("🔍 Analyzing document type..."):
                    detected_type = detect_document_type(text, uploaded_file.name)
                    # Update the doc_type in the session state
                    st.session_state.detected_doc_type = detected_type
                    doc_type = detected_type  # Update the current doc_type
                
                st.success(f"✅ Detected document type: {detected_type}")
                
                # Analyze with NVIDIA model
                st.info(f"🤖 Analyzing document with {st.session_state.model_name}...")
                analysis = analyze_with_nvidia(text, doc_type=doc_type)
                
                # Initialize document_for_chat with the full text by default
                document_for_chat = text
                
                if analysis:
                    st.success("✅ Analysis complete!")
                    display_analysis(analysis)
                else:
                    st.warning("⚠️ Could not analyze with AI. Falling back to basic summary.")
                    summary = generate_legal_summary(text)
                    st.markdown(summary)
                    document_for_chat = summary  # Use summary for chat
                
                # Store the document text for chat
                st.session_state.document_text = document_for_chat
                
                # Add chat interface after analysis
                st.markdown("---")
                st.subheader("💬 Ask questions about the document")
                
                # Initialize chat history if it doesn't exist
                if 'chat_history' not in st.session_state:
                    st.session_state.chat_history = []
                
                # Display chat history
                for message in st.session_state.chat_history:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
                
                # Chat input with unique key
                if question := st.chat_input("Ask a question about the document...", key="document_qa_chat"):
                    # Add user message to chat history
                    st.session_state.chat_history.append({"role": "user", "content": question})
                    
                    # Display user message
                    with st.chat_message("user"):
                        st.markdown(question)
                    
                    # Generate and display assistant response
                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            response = chat_about_document(
                                question=question,
                                document_text=st.session_state.document_text,
                                doc_type=doc_type
                            )
                            st.markdown(response)
                    
                    # Add assistant response to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    
                    # Rerun to update the chat interface
                    st.rerun()
            else:
                st.error("❌ Could not extract text from the document. Please try another file.")
    

if __name__ == "__main__":
    main()
