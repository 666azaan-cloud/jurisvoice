# JurisVoice - AI Legal Feedback Analyzer

## Problem Statement
**SIH 2025 Problem Statement ID: 25035**  
Sentiment analysis of comments received through E-consultation module

## Features

### Core AI Capabilities
- **Advanced Sentiment Analysis** using BERT-based transformer models
- **Automated Summary Generation** using BART
- **Word Cloud Visualization** with smart keyword extraction
- **Multi-dimensional Emotion Detection** beyond basic positive/negative

### Innovation Features (From PPT)
1. **Interactive Q&A Summarizer**: Officials can ask natural language questions like "What are the main complaints?" and get instant AI-generated answers
2. **Automated "What People Want" Insights**: Automatically generates quantifiable insights like "30% want simpler language", "15% highlight misuse risk"
3. **Real-time Analysis Dashboard** with comprehensive visualizations

## Technologies Used

### Backend & AI
- **Python**: Core programming language
- **Transformers (Hugging Face)**: BERT, RoBERTa, DistilBERT for sentiment analysis
- **BART**: For text summarization
- **NLTK & spaCy**: Text preprocessing and language processing
- **TextBlob**: Additional sentiment and emotion analysis

### Frontend & Visualization
- **Streamlit**: Interactive web application framework
- **Plotly**: Interactive charts and visualizations
- **Matplotlib**: Word cloud generation
- **Pandas**: Data manipulation and analysis

### Deployment
- **Streamlit Cloud**: Free hosting and deployment
- **GitHub**: Version control and repository management

## Installation & Setup

### 1. Clone/Download the Project
```bash
git clone <your-repo-url>
cd jurisvoice
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Application
```bash
streamlit run JurisVoice-app.py
```

### 4. Access the App
Open your browser and go to: `http://localhost:8501`

## Usage

### For Development/Testing
1. Run the app locally using the command above
2. Use "Sample Data" checkbox to test with demo comments
3. Or upload your own CSV file with comments

### For Production/Hosting
1. Push code to GitHub repository
2. Connect to Streamlit Cloud (streamlit.io/cloud)
3. Deploy with one click - app will be live on the internet

## File Structure
```
jurisvoice/
├── JurisVoice-app.py       # Main application file
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── data/
    └── sample_comments.csv # Sample data for testing
```

## Key Features Implementation

### 1. Sentiment Analysis
- Uses RoBERTa model fine-tuned for sentiment classification
- Provides confidence scores for each prediction
- Maps emotions to categories: supportive, worried, satisfied, etc.

### 2. Summary Generation
- BART model for extractive and abstractive summarization
- Handles long documents by chunking
- Generates concise summaries of all feedback

### 3. Interactive Q&A (Innovation)
```python
# Example usage in the app:
question = "What are the main complaints?"
answer = interactive_qa(question, comments, qa_model)
```

### 4. Automated Insights (Innovation)
```python
# Generates insights like:
# "⚠️ 65% of comments express concerns or objections"
# "👍 35% actively support the proposed changes"
```

### 5. Word Cloud
- Removes common stopwords
- Highlights key themes and keywords
- Customizable colors and layout

## Demo Data
The app includes sample legal consultation comments for immediate testing:
- Mix of positive, negative, and neutral feedback
- Covers typical concerns: complexity, implementation, compliance burden
- Shows various stakeholder perspectives

## Deployment Options

### Option 1: Streamlit Cloud (Recommended)
1. Create GitHub repository
2. Push your code
3. Visit streamlit.io/cloud
4. Connect GitHub repo
5. Deploy automatically

### Option 2: Heroku
1. Create Procfile: `web: streamlit run JurisVoice-app.py --server.port $PORT`
2. Deploy to Heroku

### Option 3: Local Development
- Perfect for development and testing
- Run `streamlit run JurisVoice-app.py`

## Future Enhancements
- Multi-language support (Hindi, regional languages)
- Integration with MCA21 portal APIs
- Advanced bias detection
- Real-time comment processing
- Export to various formats (PDF, Excel)

## Team: JurisVoice
**Smart India Hackathon 2025**  
Building AI-powered solutions for transparent governance

---

## Support
For issues or questions about deployment:
1. Check the terminal/console for error messages
2. Ensure all requirements are installed
3. Verify Python version compatibility (3.8+)

**Ready to revolutionize legal consultation analysis!** ⚖️🚀