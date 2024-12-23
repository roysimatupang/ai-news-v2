import os
import requests
from dotenv import load_dotenv
import streamlit as st
from typing import List, Dict
import anthropic
import re
from time import sleep
import plotly.express as px
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load environment variables
load_dotenv()

# Function to generate response using Claude API
def get_response(user_content: str, system_prompt: str) -> str:
    try:
        client = anthropic.Anthropic(api_key=os.environ.get("CLAUDE_API_KEY"))
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_content}],
        )
        return response.content
    except Exception as e:
        return f"Error: {e}"

def clean_response(response) -> str:
    """Cleans the LLM response by removing unwanted TextBlock tags and extra formatting."""
    if isinstance(response, list):  # If response is a list, join elements into a string
        response = " ".join([str(item) for item in response])
    # Replace \n with actual newlines
    response = response.replace("\\n", "\n")
    # Correct regex to handle `[TextBlock(text='...')]`
    cleaned = re.sub(r"\[TextBlock\(text=['\"](.*?)['\"],\s*type=['\"]text['\"]\)\]", r"\1", response)
    # Remove any remaining `[TextBlock(...)]` or similar patterns
    cleaned = re.sub(r"\[TextBlock\(.*?\)\]", "", cleaned)
    # Remove leading and trailing whitespace
    return cleaned.strip()


# Initialize Claude LLM using a wrapper
class LLMWrapper:
    def __init__(self):
        self.api_key = os.getenv('CLAUDE_API_KEY')
        if not self.api_key:
            raise ValueError("Claude API key is missing. Check your .env file.")

    def generate(self, user_content: str, system: str) -> str:
        return get_response(user_content=user_content, system_prompt=system)

# Base class for agents
class BaseAgent:
    def __init__(self, llm_wrapper: LLMWrapper):
        self.llm_wrapper = llm_wrapper

class NewsCrawler(BaseAgent):
    def fetch_news(self, query: str) -> List[Dict[str, str]]:
        api_key = os.getenv('NEWS_API_KEY')
        url = f"https://newsapi.org/v2/everything?q={query}&language=en&apiKey={api_key}"
        response = requests.get(url)
        if response.status_code == 200:
            articles = response.json().get("articles", [])
            return [
                {
                    "title": a.get("title", "Untitled"),
                    "description": a.get("description", "No description"),
                    "source": a.get("source", {}).get("name", "Unknown"),
                    "url": a.get("url", "#")
                }
                for a in articles if a.get("title") and a.get("description") and a.get("source", {}).get("name") != "[Removed]"
            ][:3]
        return []

class NewsAnalyst(BaseAgent):
    def analyze_articles(self, articles: List[Dict[str, str]]) -> List[Dict[str, str]]:
        analyzed_articles = []
        for article in articles:
            try:
                raw_keywords = self.llm_wrapper.generate(
                    user_content=f"Extract 3 to 5 key highlights in sentences ( numbered markdown ) and extract key words from the following text: {article['description'][:300]}.",
                    system="You are a helpful assistant extracting summary news highlights and  key words."
                )
                keywords = clean_response(str(raw_keywords))
                analyzed_articles.append({
                    "title": article["title"],
                    "description": article["description"],
                    "key_highlights": keywords,
                    "source": article["source"],
                    "url": article["url"]
                })
            except Exception as e:
                analyzed_articles.append({
                    "title": article["title"],
                    "description": article["description"],
                    "error": f"Error analyzing article: {e}"
                })
        return analyzed_articles

class BiasDetection(BaseAgent):
    def detect_bias(self, articles: List[Dict[str, str]]) -> List[Dict[str, str]]:
        bias_results = []
        for article in articles:
            try:
                raw_bias = self.llm_wrapper.generate(
                    user_content=f"Analyze the following text for bias and suggest ways to make it more balanced in 3 sentences analyzing: {article['description'][:300]}.",
                    system="You are a helpful assistant detecting bias in journalism."
                )
                bias_analysis = clean_response(str(raw_bias))
                bias_results.append({
                    "title": article["title"],
                    "bias_analysis": bias_analysis,
                    "source": article["source"],
                    "url": article["url"]
                })
            except Exception as e:
                bias_results.append({
                    "title": article["title"],
                    "error": f"Error detecting bias: {e}"
                })
        return bias_results

class WordCloudPlotter(BaseAgent):
    def plot_word_cloud(self, analyzed_articles: List[Dict[str, str]]):
        all_keywords = " ".join([article["key_highlights"] for article in analyzed_articles if "key_highlights" in article])
        # Exclude unwanted words from word cloud
        all_keywords = re.sub(r"\b(TextBlock|Text|Key|extracted)\b", "", all_keywords, flags=re.IGNORECASE)
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_keywords)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title("Key Words Word Cloud", fontsize=20)
        st.pyplot(plt)

# Main App Class
class NewsSummarizerApp:
    def __init__(self):
        self.llm_wrapper = LLMWrapper()
        self.crawler = NewsCrawler(self.llm_wrapper)
        self.analyst = NewsAnalyst(self.llm_wrapper)
        self.bias_detector = BiasDetection(self.llm_wrapper)
        self.plotter = WordCloudPlotter(self.llm_wrapper)

    def run(self):
        st.title("AI-Powered News Analysis with Bias Detection")

        # Input
        query = st.text_input("Enter a topic to search for news:")

        if query:
            with st.spinner("Fetching news..."):
                articles = self.crawler.fetch_news(query)

            if not articles:
                st.error("No articles found.")
                return

            st.success("News articles fetched successfully.")
            st.write("### Fetched Articles")
            for article in articles:
                st.markdown(f"- **[{article['title']}]({article['url']})**: {article['description']} (Source: {article['source']})")

            with st.spinner("Analyzing articles..."):
                for _ in range(5):  # Simulating a loading bar
                    sleep(0.2)
                analyzed_articles = self.analyst.analyze_articles(articles)

            st.success("Articles analyzed successfully.")
            st.write("### Analyzed Articles")
            for article in analyzed_articles:
                if 'error' in article:
                    st.markdown(f"- **{article['title']}**: Error - {article['error']}")
                else:
                    st.markdown(f"**[{article['title']}]({article['url']})**")
                    st.markdown(f"<p style='text-align: center; font-size: x-large; font-weight: bold;'>News Highlights:</p>\n<ul><li>{article['key_highlights']}</li></ul>", unsafe_allow_html=True)

            with st.spinner("Detecting bias..."):
                for _ in range(5):  # Simulating a loading bar
                    sleep(0.2)
                bias_results = self.bias_detector.detect_bias(analyzed_articles)

            st.success("Bias detection completed.")
            st.write("### Bias Detection Results")
            for result in bias_results:
                if 'error' in result:
                    st.markdown(f"- **{result['title']}**: Error - {result['error']}")
                else:
                    st.markdown(f"**[{result['title']}]({result['url']})**")
                    st.markdown(f"<p style='text-align: center; font-size: x-large; font-weight: bold;'>Bias Analysis:</p>\n<p>{result['bias_analysis']}</p>", unsafe_allow_html=True)

            st.write("### Key Words Word Cloud")
            self.plotter.plot_word_cloud(analyzed_articles)

if __name__ == "__main__":
    app = NewsSummarizerApp()
    try:
        #st.write("<p style='font-size: x-large; font-weight: bold;'>Testing LLM</p>", unsafe_allow_html=True)

        #response = app.llm_wrapper.generate(user_content="Test connection.", system="You are a system tester.")
        #st.markdown(f"<p style='font-size: x-large; font-weight: bold;'>LLM Response:</p>\n<p style='font-size: large;'>{clean_response(response)}</p>", unsafe_allow_html=True)
        app.run()
    except Exception as e:
        st.error(f"Application startup error: {e}")
