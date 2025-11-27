import threading
import time
import logging
from web_crawler import WebScraper
from serper_service import SerperClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WebContentFetcher:
    def __init__(self, query):
        # Initialize the fetcher with a search query
        self.query = query
        self.web_contents = []  # Stores the fetched web contents
        self.error_urls = []  # Stores URLs that resulted in an error during fetching
        self.web_contents_lock = threading.Lock()  # Lock for thread-safe operations on web_contents
        self.error_urls_lock = threading.Lock()  # Lock for thread-safe operations on error_urls

    def _web_crawler_thread(self, thread_id: int, urls: list):
        # Thread function to crawl each URL
        try:
            logger.info(f"Thread {thread_id}: Starting web crawler")
            start_time = time.time()

            url = urls[thread_id]
            logger.debug(f"Thread {thread_id}: Crawling {url}")
            scraper = WebScraper()
            content = scraper.scrape_url(url, 0)

            logger.debug(f"Thread {thread_id}: Scraped {len(content)} characters from {url}")

            # If the scraped content is too short, try extending the crawl rules
            if 0 < len(content) < 800:
                logger.info(f"Thread {thread_id}: Content too short ({len(content)} chars), retrying with extended rules")
                content = scraper.scrape_url(url, 1)
                logger.debug(f"Thread {thread_id}: Retry scraped {len(content)} characters")

            # If the content length is sufficient, add it to the shared list
            if len(content) > 300:
                with self.web_contents_lock:
                    self.web_contents.append({"url": url, "content": content})
                logger.info(f"Thread {thread_id}: Successfully added content ({len(content)} chars) from {url}")
            else:
                logger.warning(f"Thread {thread_id}: Content too short ({len(content)} chars), skipping {url}")

            end_time = time.time()
            logger.info(f"Thread {thread_id}: Completed in {end_time - start_time:.2f}s")

        except Exception as e:
            # Handle any exceptions, log the error, and store the URL
            with self.error_urls_lock:
                self.error_urls.append(url)
            logger.error(f"Thread {thread_id}: Error crawling {url}: {e}", exc_info=True)

    def _serper_launcher(self):
        # Function to launch the Serper client and get search results
        logger.info(f"Launching Serper search for query: '{self.query}'")
        serper_client = SerperClient()
        serper_results = serper_client.serper(self.query)
        components = serper_client.extract_components(serper_results)
        logger.info(f"Serper returned {components.get('count', 0)} results")
        return components

    def _crawl_threads_launcher(self, url_list):
        # Create and start threads for each URL in the list
        logger.info(f"Starting {len(url_list)} crawler threads")
        threads = []
        for i in range(len(url_list)):
            thread = threading.Thread(target=self._web_crawler_thread, args=(i, url_list))
            threads.append(thread)
            thread.start()

        # Wait for all threads to finish execution
        logger.info("Waiting for all crawler threads to complete...")
        for thread in threads:
            thread.join()

        logger.info(f"All threads completed. Successfully crawled {len(self.web_contents)} URLs, {len(self.error_urls)} errors")

    def fetch(self):
        # Main method to fetch web content based on the query
        logger.info(f"Starting fetch for query: '{self.query}'")
        serper_response = self._serper_launcher()

        if not serper_response:
            logger.error("Serper response is None or empty")
            return [], None

        if not serper_response.get("links"):
            logger.error("No links found in Serper response")
            logger.debug(f"Serper response: {serper_response}")
            return [], None

        url_list = serper_response["links"]
        logger.info(f"Found {len(url_list)} URLs to crawl")

        self._crawl_threads_launcher(url_list)

        # Reorder the fetched content to match the order of URLs
        ordered_contents = [
            next((item['content'] for item in self.web_contents if item['url'] == url), '')
            for url in url_list
        ]

        non_empty_count = sum(1 for content in ordered_contents if content)
        logger.info(f"Fetch completed: {non_empty_count}/{len(ordered_contents)} URLs returned content")

        if non_empty_count == 0:
            logger.warning("No content was successfully fetched from any URL")

        return ordered_contents, serper_response

# Example usage
if __name__ == "__main__":
    fetcher = WebContentFetcher("What happened to Silicon Valley Bank")
    contents, serper_response = fetcher.fetch()

    print(serper_response)
    print(contents, '\n\n')
    