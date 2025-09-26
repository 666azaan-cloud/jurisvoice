import streamlit as st
import pandas as pd
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from collections import Counter
import re
import nltk
from textblob import TextBlob
import io
import base64
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Page config
st.set_page_config(
    page_title="JurisVoice - AI Legal Feedback Analyzer",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1F497D;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #403152;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .insight-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1F497D;
        margin: 1rem 0;
    }
    .qa-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #403152;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analyzed_data' not in st.session_state:
    st.session_state.analyzed_data = None
if 'qa_history' not in st.session_state:
    st.session_state.qa_history = []

@st.cache_resource
def load_models():
    """Load all AI models with caching for better performance"""
    try:
        # Load sentiment analysis model (BERT-based)
        sentiment_model = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest"
        )
        
        # Load summarization model
        summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            max_length=150,
            min_length=50,
            do_sample=False
        )
        
        # Load question-answering model for Q&A feature
        qa_model = pipeline(
            "question-answering",
            model="distilbert-base-cased-distilled-squad"
        )
        
        return sentiment_model, summarizer, qa_model
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None

def preprocess_text(text):
    """Clean and preprocess text data"""
    if pd.isna(text) or text == "":
        return ""
    
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.\,\!\?]', ' ', text)
    
    # Remove extra whitespaces
    text = ' '.join(text.split())
    
    return text

def analyze_sentiment_advanced(comments, sentiment_model):
    """Advanced sentiment analysis with emotion detection"""
    results = []
    
    for comment in comments:
        if not comment or pd.isna(comment):
            continue
            
        # Basic sentiment using transformer
        try:
            sentiment_result = sentiment_model(comment[:512])  # Limit token length
            sentiment = sentiment_result[0]
            
            # Map labels to standard format
            if sentiment['label'] in ['LABEL_2', 'positive', 'POSITIVE']:
                mapped_sentiment = 'positive'
            elif sentiment['label'] in ['LABEL_0', 'negative', 'NEGATIVE']:
                mapped_sentiment = 'negative'
            else:
                mapped_sentiment = 'neutral'
                
            # Additional emotion analysis using TextBlob
            blob = TextBlob(comment)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Emotion categorization
            if polarity > 0.1:
                emotion = 'supportive' if subjectivity > 0.5 else 'satisfied'
            elif polarity < -0.1:
                emotion = 'worried' if subjectivity > 0.5 else 'dissatisfied'
            else:
                emotion = 'neutral'
            
            results.append({
                'comment': comment,
                'sentiment': mapped_sentiment,
                'confidence': sentiment['score'],
                'polarity': polarity,
                'subjectivity': subjectivity,
                'emotion': emotion
            })
            
        except Exception as e:
            st.warning(f"Error analyzing comment: {str(e)[:100]}")
            continue
    
    return results

def generate_summary(comments, summarizer):
    """Generate comprehensive summary of all comments"""
    try:
        # Combine all comments
        all_text = " ".join([str(comment) for comment in comments if comment and not pd.isna(comment)])
        
        # Split into chunks if text is too long
        max_chunk_length = 1000
        chunks = [all_text[i:i+max_chunk_length] for i in range(0, len(all_text), max_chunk_length)]
        
        summaries = []
        for chunk in chunks[:3]:  # Limit to first 3 chunks for performance
            if len(chunk.strip()) > 100:
                try:
                    summary = summarizer(chunk, max_length=150, min_length=30, do_sample=False)
                    summaries.append(summary[0]['summary_text'])
                except:
                    continue
        
        return " ".join(summaries) if summaries else "Unable to generate summary from the provided comments."
    
    except Exception as e:
        return f"Summary generation failed: {str(e)}"

def generate_wordcloud(comments):
    """Generate word cloud from comments"""
    try:
        # Combine all comments
        text = " ".join([str(comment) for comment in comments if comment and not pd.isna(comment)])
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        stop_words.update(['law', 'draft', 'section', 'provision', 'would', 'could', 'should'])
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            stopwords=stop_words,
            max_words=100,
            colormap='viridis'
        ).generate(text)
        
        return wordcloud
    except Exception as e:
        st.error(f"Error generating word cloud: {str(e)}")
        return None

def generate_insights(sentiment_data):
    """Generate automated 'What People Want' insights"""
    if not sentiment_data:
        return []
    
    df = pd.DataFrame(sentiment_data)
    total_comments = len(df)
    
    insights = []
    
    # Sentiment distribution insights
    sentiment_counts = df['sentiment'].value_counts(normalize=True) * 100
    
    if 'negative' in sentiment_counts and sentiment_counts['negative'] > 30:
        insights.append(f"⚠️ {sentiment_counts['negative']:.0f}% of comments express concerns or objections")
    
    if 'positive' in sentiment_counts:
        insights.append(f"✅ {sentiment_counts['positive']:.0f}% of comments show support for the draft")
    
    # Emotion-based insights
    emotion_counts = df['emotion'].value_counts(normalize=True) * 100
    
    if 'worried' in emotion_counts and emotion_counts['worried'] > 20:
        insights.append(f"😰 {emotion_counts['worried']:.0f}% of comments indicate worry about implementation")
    
    if 'supportive' in emotion_counts and emotion_counts['supportive'] > 25:
        insights.append(f"👍 {emotion_counts['supportive']:.0f}% actively support the proposed changes")
    
    # Confidence level insights
    high_conf_negative = df[(df['sentiment'] == 'negative') & (df['confidence'] > 0.8)]
    if len(high_conf_negative) > total_comments * 0.15:
        insights.append(f"🔴 {len(high_conf_negative)/total_comments*100:.0f}% express strong disagreement")
    
    # Subjectivity insights
    highly_subjective = df[df['subjectivity'] > 0.7]
    if len(highly_subjective) > total_comments * 0.4:
        insights.append(f"💭 {len(highly_subjective)/total_comments*100:.0f}% of comments are opinion-based rather than factual")
    
    return insights

def interactive_qa(question, comments, qa_model):
    """Interactive Q&A feature for officials"""
    try:
        # Combine all comments as context
        context = " ".join([str(comment) for comment in comments if comment and not pd.isna(comment)][:50])
        
        if len(context) < 50:
            return "Not enough context available for Q&A analysis."
        
        # Truncate context if too long
        context = context[:2000]
        
        result = qa_model(question=question, context=context)
        
        confidence = result['score']
        answer = result['answer']
        
        if confidence > 0.3:
            return f"{answer} (Confidence: {confidence:.2f})"
        else:
            return "Unable to find a confident answer in the provided comments."
            
    except Exception as e:
        return f"Q&A analysis failed: {str(e)}"

def main():
    # Header
    st.markdown('<h1 class="main-header">⚖️ JurisVoice - AI Legal Feedback Analyzer</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666; font-size: 1.2rem;">Smart India Hackathon 2025 | Problem Statement 25035</p>', unsafe_allow_html=True)
    
    # Load models
    with st.spinner("Loading AI models... This may take a moment."):
        sentiment_model, summarizer, qa_model = load_models()
    
    if not all([sentiment_model, summarizer, qa_model]):
        st.error("Failed to load required AI models. Please refresh the page.")
        return
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 Analysis Options")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload Comments File (CSV)",
            type=['csv'],
            help="Upload a CSV file with comments to analyze"
        )
        
        # Sample data option
        use_sample = st.checkbox("Use Sample Data", help="Try the system with sample legal comments")
        
        st.markdown("---")
        
        # Innovation features
        st.markdown("### 🚀 Innovation Features")
        st.markdown("""
        - **Interactive Q&A Summarizer**
        - **Automated Insights Generation**
        - **Multi-dimensional Sentiment Analysis**
        - **Real-time Word Cloud Generation**
        """)
    
    # Main content
    if uploaded_file is not None or use_sample:
        
        # Load data
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                comments = df.iloc[:, 0].tolist()  # Assume first column contains comments
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
                return
        else:
            # Sample data for demonstration
            comments = [
                "This draft law is too complex and needs simpler language for common citizens to understand.",
                "I strongly support this amendment as it will improve transparency in corporate governance.",
                "The proposed section 12A might be misused by authorities. Please reconsider this provision.",
                "Excellent initiative by MCA. This will streamline business processes significantly.",
                "I am worried about the compliance burden on small businesses. Please provide more exemptions.",
                "The timeline for implementation is too short. Extend it to at least 12 months.",
                "This law will benefit the economy and improve ease of doing business in India.",
                "Some clauses are ambiguous and may lead to legal disputes. Please clarify.",
                "I appreciate the government's effort to modernize these regulations.",
                "The penalties mentioned are too harsh for minor violations.",
            ]
        
        st.success(f"📁 Loaded {len(comments)} comments for analysis")
        
        # Perform analysis
        with st.spinner("Analyzing comments with AI models..."):
            
            # Preprocess comments
            cleaned_comments = [preprocess_text(comment) for comment in comments]
            
            # Sentiment analysis
            sentiment_data = analyze_sentiment_advanced(cleaned_comments, sentiment_model)
            
            # Generate summary
            summary = generate_summary(cleaned_comments, summarizer)
            
            # Generate word cloud
            wordcloud = generate_wordcloud(cleaned_comments)
            
            # Generate insights
            insights = generate_insights(sentiment_data)
            
            st.session_state.analyzed_data = {
                'sentiment_data': sentiment_data,
                'summary': summary,
                'wordcloud': wordcloud,
                'insights': insights,
                'comments': cleaned_comments
            }
        
        # Display results
        if st.session_state.analyzed_data:
            
            # Summary Section
            st.markdown('<h2 class="sub-header">📄 Executive Summary</h2>', unsafe_allow_html=True)
            st.markdown(f'<div class="insight-box">{st.session_state.analyzed_data["summary"]}</div>', unsafe_allow_html=True)
            
            # Key Metrics
            st.markdown('<h2 class="sub-header">📊 Key Metrics</h2>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            sentiment_df = pd.DataFrame(st.session_state.analyzed_data['sentiment_data'])
            
            with col1:
                total_comments = len(sentiment_df)
                st.metric("Total Comments", total_comments)
            
            with col2:
                positive_pct = (sentiment_df['sentiment'] == 'positive').mean() * 100
                st.metric("Positive Sentiment", f"{positive_pct:.1f}%")
            
            with col3:
                negative_pct = (sentiment_df['sentiment'] == 'negative').mean() * 100
                st.metric("Negative Sentiment", f"{negative_pct:.1f}%")
            
            with col4:
                avg_confidence = sentiment_df['confidence'].mean()
                st.metric("Avg Confidence", f"{avg_confidence:.2f}")
            
            # Automated Insights (Innovation Feature)
            st.markdown('<h2 class="sub-header">🔍 Automated "What People Want" Insights</h2>', unsafe_allow_html=True)
            
            for insight in st.session_state.analyzed_data['insights']:
                st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Sentiment Distribution
                st.markdown("### Sentiment Distribution")
                sentiment_counts = sentiment_df['sentiment'].value_counts()
                fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index, 
                           color_discrete_map={'positive': '#28a745', 'negative': '#dc3545', 'neutral': '#ffc107'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Emotion Analysis
                st.markdown("### Emotion Analysis")
                emotion_counts = sentiment_df['emotion'].value_counts()
                fig2 = px.bar(x=emotion_counts.index, y=emotion_counts.values, 
                             color=emotion_counts.values, color_continuous_scale='viridis')
                st.plotly_chart(fig2, use_container_width=True)
            
            with col2:
                # Word Cloud
                st.markdown("### Word Cloud")
                if st.session_state.analyzed_data['wordcloud']:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.imshow(st.session_state.analyzed_data['wordcloud'], interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
                else:
                    st.error("Could not generate word cloud")
                
                # Confidence vs Sentiment
                st.markdown("### Confidence vs Sentiment")
                fig3 = px.scatter(sentiment_df, x='confidence', y='polarity', 
                                color='sentiment', size='subjectivity',
                                color_discrete_map={'positive': '#28a745', 'negative': '#dc3545', 'neutral': '#ffc107'})
                st.plotly_chart(fig3, use_container_width=True)
            
            # Interactive Q&A Feature (Innovation)
            st.markdown('<h2 class="sub-header">🤖 Interactive Q&A Summarizer</h2>', unsafe_allow_html=True)
            st.markdown("Ask natural language questions about the feedback and get instant AI-generated answers.")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                question = st.text_input(
                    "Ask a question about the feedback:",
                    placeholder="e.g., 'What are the main complaints?', 'What do people like most?'"
                )
            
            with col2:
                if st.button("Ask AI", type="primary"):
                    if question:
                        with st.spinner("Generating answer..."):
                            answer = interactive_qa(question, st.session_state.analyzed_data['comments'], qa_model)
                            st.session_state.qa_history.append({
                                'question': question,
                                'answer': answer,
                                'timestamp': datetime.now().strftime("%H:%M:%S")
                            })
            
            # Display Q&A History
            if st.session_state.qa_history:
                st.markdown("### Recent Q&A")
                for i, qa in enumerate(reversed(st.session_state.qa_history[-5:])):  # Show last 5
                    with st.expander(f"Q: {qa['question']} ({qa['timestamp']})"):
                        st.markdown(f"**A:** {qa['answer']}")
            
            # Detailed Comments Table
            st.markdown('<h2 class="sub-header">📋 Detailed Analysis</h2>', unsafe_allow_html=True)
            
            # Create detailed DataFrame for display
            display_df = sentiment_df[['comment', 'sentiment', 'emotion', 'confidence', 'polarity']].copy()
            display_df['confidence'] = display_df['confidence'].round(3)
            display_df['polarity'] = display_df['polarity'].round(3)
            display_df.columns = ['Comment', 'Sentiment', 'Emotion', 'Confidence', 'Polarity']
            
            st.dataframe(display_df, use_container_width=True)
            
            # Export functionality
            st.markdown("### 💾 Export Results")
            col1, col2 = st.columns(2)
            
            with col1:
                # CSV export
                csv = display_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV Report",
                    data=csv,
                    file_name=f"jurisvoice_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Summary report export
                report = f"""
JurisVoice Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

EXECUTIVE SUMMARY:
{st.session_state.analyzed_data['summary']}

KEY INSIGHTS:
{chr(10).join(f'• {insight}' for insight in st.session_state.analyzed_data['insights'])}

STATISTICS:
• Total Comments: {len(sentiment_df)}
• Positive Sentiment: {positive_pct:.1f}%
• Negative Sentiment: {negative_pct:.1f}%
• Average Confidence: {avg_confidence:.2f}
                """
                
                st.download_button(
                    label="Download Summary Report",
                    data=report,
                    file_name=f"jurisvoice_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
    
    else:
        # Landing page
        st.markdown("""
        ### 🚀 Welcome to JurisVoice
        
        **A comprehensive AI-powered platform for automated sentiment analysis and insight generation from public feedback on legal documents.**
        
        #### Key Features:
        - **Advanced Sentiment Analysis** using BERT-based models
        - **Automated Summary Generation** using BART
        - **Interactive Q&A Summarizer** for natural language queries
        - **Real-time Word Cloud Generation**
        - **Automated "What People Want" Insights**
        - **Multi-dimensional Emotion Detection**
        
        #### Innovation Highlights:
        1. **Interactive Q&A Summarizer**: Officials can ask questions like "What are the main complaints?" and get instant AI-generated answers
        2. **Automated Insights**: System automatically generates quantifiable insights like "30% want simpler language"
        3. **Multi-dimensional Analysis**: Beyond basic sentiment, analyzes emotions, subjectivity, and confidence levels
        
        **👈 Upload a CSV file or use sample data to get started!**
        """)
        
        # Demo section
        with st.expander("🎯 See Sample Analysis"):
            st.image("https://via.placeholder.com/800x400/1F497D/FFFFFF?text=JurisVoice+Dashboard+Preview", 
                    caption="Dashboard Preview")

if __name__ == "__main__":
    main()