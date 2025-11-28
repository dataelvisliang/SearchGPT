import requests
import re
import io
import logging
from bs4 import BeautifulSoup
try:
    from PyPDF2 import PdfReader
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

# Configure logging
logger = logging.getLogger(__name__)

class WebScraper:
    def __init__(self, user_agent='macOS'):
        # Initialize the scraper with a user agent (default is 'macOS')
        self.headers = self._get_headers(user_agent)

    def _get_headers(self, user_agent):
        # Private method to get headers for the request based on the specified user agent
        if user_agent == 'macOS':
            # Headers for macOS user agent
            return {
                'Upgrade-Insecure-Requests': '1',
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
                'sec-ch-ua': '"Not/A)Brand";v="99", "Google Chrome";v="115", "Chromium";v="115"',
                'sec-ch-ua-mobile': '?0',
                'sec-ch-ua-platform': '"macOS"',
            }
        else:
            # Headers for Windows user agent
            return {
                'Upgrade-Insecure-Requests': '1',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
                'sec-ch-ua': '"Not.A/Brand";v="8", "Chromium";v="114", "Google Chrome";v="114"',
                'sec-ch-ua-mobile': '?0',
                'sec-ch-ua-platform': '"Windows"',
            }

    def get_webpage_html(self, url):
        # Fetch the HTML content of a webpage from a given URL
        response = requests.Response()  # Create an empty Response object

        try:
            # Attempt to get the webpage content with specified headers and timeout
            response = requests.get(url, headers=self.headers, timeout=15)
            response.encoding = "utf-8"
        except requests.exceptions.Timeout:
            # Add timeout exception handling here
            return response

        return response

    def convert_html_to_soup(self, html):
        # Convert the HTML string to a BeautifulSoup object for parsing
        html_string = html.text
        return BeautifulSoup(html_string, "lxml")

    def extract_main_content(self, html_soup, rule=0):
        # Extract the main content from a BeautifulSoup object
        main_content = []
        tag_rule = re.compile("^(h[1-6]|p|div)" if rule == 1 else "^(h[1-6]|p)")
        # Iterate through specified tags and collect their text
        for tag in html_soup.find_all(tag_rule):
            tag_text = tag.get_text().strip()
            if tag_text and len(tag_text.split()) > 10:
                main_content.append(tag_text)
        return "\n".join(main_content).strip()

    def extract_pdf_content(self, pdf_response):
        # Extract text content from a PDF response
        if not PDF_SUPPORT:
            logger.warning("PyPDF2 not available, skipping PDF extraction")
            return ""

        try:
            logger.info("Extracting content from PDF")
            pdf_file = io.BytesIO(pdf_response.content)
            pdf_reader = PdfReader(pdf_file)

            text_content = []
            # Extract text from all pages
            for i, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text_content.append(page_text)

            full_text = "\n".join(text_content).strip()
            logger.info(f"Successfully extracted {len(full_text)} characters from PDF ({len(pdf_reader.pages)} pages)")
            return full_text
        except Exception as e:
            # If PDF extraction fails, return empty string
            logger.error(f"Failed to extract PDF content: {e}")
            return ""

    def scrape_url(self, url, rule=0):
        # Public method to scrape a URL and extract its main content
        webpage_html = self.get_webpage_html(url)

        # Check if it's a PDF by URL or content type
        if url.endswith(".pdf") or (hasattr(webpage_html, 'headers') and webpage_html.headers.get('Content-Type', '').startswith('application/pdf')):
            logger.info(f"Detected PDF URL: {url}")
            if PDF_SUPPORT:
                return self.extract_pdf_content(webpage_html)
            else:
                logger.warning("PDF detected but PyPDF2 not available")
                return ""  # Skip if PyPDF2 not available

        soup = self.convert_html_to_soup(webpage_html)
        main_content = self.extract_main_content(soup, rule)
        return main_content

# Example usage
if __name__ == "__main__":
    scraper = WebScraper(user_agent='macOS')
    test_url = "https://en.wikipedia.org/wiki/Apple_Inc."
    main_content = scraper.scrape_url(test_url)
    print(main_content)
    