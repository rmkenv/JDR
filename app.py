import csv
import requests
import zipfile
import io
import os
import re
import tempfile
import pandas as pd
import streamlit as st
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from datetime import datetime

class WebScraperAgent:
    def __init__(self, csv_file, keywords, output_folder=None):
        """
        Initialize the web scraper agent.

        Args:
            csv_file: CSV file object containing URLs
            keywords (list): List of keywords or phrases to search for
            output_folder (str): Folder to save downloaded files
        """
        self.csv_file = csv_file
        self.keywords = keywords
        self.output_folder = output_folder or tempfile.mkdtemp()
        self.results = []

        # Create output folder if it doesn't exist
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def read_csv(self):
        """Read URLs from CSV file."""
        df = pd.read_csv(self.csv_file)

        # Find column with URLs
        url_col = None
        for col in df.columns:
            if 'url' in col.lower():
                url_col = col
                break

        # If no column with 'url' in name is found, use the first column
        if url_col is None and len(df.columns) > 0:
            url_col = df.columns[0]

        if url_col:
            return df[url_col].dropna().tolist()
        return []

    def download_file(self, url):
        """Download file from URL."""
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            st.error(f"Error downloading {url}: {e}")
            return None

    def extract_zip(self, content, url):
        """Extract zip file and search for keywords."""
        filename = os.path.basename(urlparse(url).path) or f"download_{datetime.now().strftime('%Y%m%d%H%M%S')}.zip"
        zip_path = os.path.join(self.output_folder, filename)

        # Save zip file
        with open(zip_path, 'wb') as f:
            f.write(content)

        matches = []
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Extract all files
                extract_folder = os.path.join(self.output_folder, os.path.splitext(filename)[0])
                if not os.path.exists(extract_folder):
                    os.makedirs(extract_folder)
                zip_ref.extractall(extract_folder)

                # Search for keywords in extracted files
                for file_info in zip_ref.infolist():
                    if not file_info.is_dir():
                        try:
                            file_content = zip_ref.read(file_info.filename).decode('utf-8', errors='ignore')
                            for keyword in self.keywords:
                                if keyword.lower() in file_content.lower():
                                    matches.append({
                                        'keyword': keyword,
                                        'file': file_info.filename,
                                        'context': self.get_context(file_content, keyword)
                                    })
                        except Exception as e:
                            st.warning(f"Error processing file {file_info.filename}: {e}")

            return matches
        except zipfile.BadZipFile:
            st.warning(f"Error: {zip_path} is not a valid zip file")
            return []

    def get_context(self, content, keyword, context_size=100):
        """Get context around the keyword match."""
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        matches = pattern.finditer(content)
        contexts = []

        for match in matches:
            start = max(0, match.start() - context_size)
            end = min(len(content), match.end() + context_size)
            context = content[start:end]
            contexts.append(f"...{context}...")

            # Limit to first 3 contexts
            if len(contexts) >= 3:
                break

        return contexts

    def search_webpage(self, url, response):
        """Search for keywords directly on the webpage."""
        matches = []
        try:
            content = response.text
            soup = BeautifulSoup(content, 'html.parser')
            text_content = soup.get_text()

            for keyword in self.keywords:
                if keyword.lower() in text_content.lower():
                    matches.append({
                        'keyword': keyword,
                        'file': 'webpage',
                        'context': self.get_context(text_content, keyword)
                    })

            # Look for zip file links
            zip_links = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                if href.endswith('.zip'):
                    # Handle relative URLs
                    if not href.startswith(('http://', 'https://')):
                        base_url = url.rstrip('/')
                        if href.startswith('/'):
                            href = f"{urlparse(base_url).scheme}://{urlparse(base_url).netloc}{href}"
                        else:
                            href = f"{base_url}/{href}"
                    zip_links.append(href)

            return matches, zip_links
        except Exception as e:
            st.warning(f"Error searching webpage {url}: {e}")
            return [], []

    def process_url(self, url, progress_bar=None):
        """Process a single URL."""
        result = {
            'url': url,
            'matches': [],
            'error': None
        }

        try:
            response = self.download_file(url)
            if not response:
                result['error'] = "Failed to download"
                return result

            content_type = response.headers.get('Content-Type', '')

            # Handle zip files
            if 'application/zip' in content_type or url.endswith('.zip'):
                matches = self.extract_zip(response.content, url)
                result['matches'].extend(matches)

            # Handle webpages
            else:
                webpage_matches, zip_links = self.search_webpage(url, response)
                result['matches'].extend(webpage_matches)

                # Process any zip links found on the page
                for zip_url in zip_links:
                    zip_response = self.download_file(zip_url)
                    if zip_response:
                        zip_matches = self.extract_zip(zip_response.content, zip_url)
                        for match in zip_matches:
                            match['source'] = zip_url
                        result['matches'].extend(zip_matches)

        except Exception as e:
            result['error'] = str(e)
            st.error(f"Error processing {url}: {e}")

        return result

    def run(self, progress_callback=None):
        """Run the agent on all URLs in the CSV."""
        urls = self.read_csv()

        if not urls:
            st.warning("No URLs found in the CSV file.")
            return []

        for i, url in enumerate(urls):
            result = self.process_url(url)
            self.results.append(result)

            if progress_callback:
                progress_callback((i + 1) / len(urls))

        return self.results

    def get_results_df(self):
        """Convert results to a pandas DataFrame."""
        data = []

        for result in self.results:
            url = result['url']
            if result['matches']:
                for match in result['matches']:
                    for context in match['context']:
                        data.append({
                            'URL': url,
                            'Keyword': match['keyword'],
                            'File': match.get('file', ''),
                            'Context': context
                        })
            else:
                data.append({
                    'URL': url,
                    'Keyword': 'NO MATCHES',
                    'File': '',
                    'Context': ''
                })

        return pd.DataFrame(data)

# Streamlit app
def main():
    st.set_page_config(page_title="URL Processor & Keyword Finder", layout="wide")

    st.title("URL Processor & Keyword Finder")
    st.write("""
    This app processes URLs from a CSV file, downloads and unzips files, and searches for keywords.
    """)

    # File upload
    st.header("Step 1: Upload CSV File with URLs")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Display CSV preview
        df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded CSV:")
        st.dataframe(df.head())

        # Keywords input
        st.header("Step 2: Enter Keywords to Search For")
        keywords_input = st.text_area("Enter keywords or phrases (one per line):",
                                     "example\ntest\ndata")

        keywords = [k.strip() for k in keywords_input.split('\n') if k.strip()]

        if st.button("Process URLs and Find Keywords"):
            if not keywords:
                st.error("Please enter at least one keyword.")
            else:
                # Reset the file pointer to the beginning
                uploaded_file.seek(0)

                # Create progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Create agent and run
                status_text.text("Initializing...")
                agent = WebScraperAgent(uploaded_file, keywords)

                status_text.text("Processing URLs...")

                # Define progress callback
                def update_progress(progress):
                    progress_bar.progress(progress)
                    status_text.text(f"Processing URLs... {int(progress * 100)}%")

                # Run the agent
                results = agent.run(update_progress)

                # Show results
                status_text.text("Processing complete!")
                progress_bar.progress(1.0)

                st.header("Results")

                # Get results as DataFrame
                results_df = agent.get_results_df()

                if not results_df.empty:
                    st.write(f"Found {len(results_df)} matches across {len(results)} URLs")
                    st.dataframe(results_df)

                    # Download button for results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name="keyword_search_results.csv",
                        mime="text/csv"
                    )
                else:
                    st.write("No matches found.")

if __name__ == "__main__":
    main()

# Created/Modified files during execution:
# - Temporary files in system temp directory
# - keyword_search_results.csv (when downloaded by user)
