import requests
import re
import json
import yaml
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SerperClient:
    def __init__(self):
        # Load configuration from config.yaml file
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.yaml')
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        # Set up the URL and headers for the Serper API
        self.url = "https://google.serper.dev/search"
        self.headers = {
            "X-API-KEY": config["serper_api_key"],  # API key from config file
            "Content-Type": "application/json"
        }

    def serper(self, query: str):
        # Configure the query parameters for Serper API
        serper_settings = {"q": query}

        # Check if the query contains Chinese characters and adjust settings accordingly
        if self._contains_chinese(query):
            serper_settings.update({"gl": "cn", "hl": "zh-cn",})
            logger.info(f"Chinese query detected. Settings: {serper_settings}")

        payload = json.dumps(serper_settings)
        logger.info(f"Sending Serper API request for query: '{query}'")

        try:
            # Perform the POST request to the Serper API and return the JSON response
            response = requests.request("POST", self.url, headers=self.headers, data=payload, timeout=30)
            response.raise_for_status()
            result = response.json()

            logger.info(f"Serper API response received. Status: {response.status_code}")
            logger.debug(f"Response keys: {result.keys()}")

            if 'organic' in result:
                logger.info(f"Found {len(result.get('organic', []))} organic results")
            else:
                logger.warning("No 'organic' key in Serper response")
                logger.debug(f"Full response: {result}")

            return result

        except requests.exceptions.RequestException as e:
            logger.error(f"Serper API request failed: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Serper response as JSON: {e}")
            raise

    def _contains_chinese(self, query: str):
        # Check if a string contains Chinese characters using a regular expression
        pattern = re.compile(r'[\u4e00-\u9fff]+')
        return bool(pattern.search(query))

    def extract_components(self, serper_response: dict):
        # Initialize lists to store the extracted components
        titles, links, snippets = [], [], []

        logger.info("Extracting components from Serper response")

        # Iterate through the 'organic' section of the response and extract information
        organic_results = serper_response.get("organic", [])
        logger.info(f"Processing {len(organic_results)} organic results")

        for i, item in enumerate(organic_results):
            title = item.get("title", "")
            link = item.get("link", "")
            snippet = item.get("snippet", "")

            titles.append(title)
            links.append(link)
            snippets.append(snippet)

            logger.debug(f"Result {i+1}: {title[:50]}... - {link}")

        # Retrieve additional information from the response
        query = serper_response.get("searchParameters", {}).get("q", "")
        count = len(links)
        language = "zh-cn" if self._contains_chinese(query) else "en-us"

        logger.info(f"Extracted {count} results in {language} language")

        # Organize the extracted data into a dictionary and return
        output_dict = {
            'query': query,
            'language': language,
            'count': count,
            'titles': titles,
            'links': links,
            'snippets': snippets
        }

        return output_dict

# Usage example
if __name__ == "__main__":    
    client = SerperClient()
    query = "What happened to Silicon Valley Bank"
    response = client.serper(query)
    components = client.extract_components(response)
    print(components)
