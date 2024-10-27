import requests
from typing import Union, Dict, Tuple, Any, List
import json

class SerpAPIClient:
    """
    A Client to interact with SERp API for performing search queries
    """
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://serpapi.com/search.json"

    def __call__(self, query: str, engine: str = "google", location: str = "")-> Union[Dict[str, Any], Tuple[int, str]]:
        """
        Perform Google Search using SERP API
        :param query: The search query string
        :param engine: The search engine to use
        :param location: The location for the search query
        :return: The search result as a JSON dictionary if successful, or a tuple containing the HTTP status code and
        error message if the request fails
        """
        params = {
            "q": query,
            "location": location,
            "hl": "en",
            "gl": "us",
            "google_domain": "google.com",
            "api_key": self.api_key
        }

        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Request to SERP API failed: {e}")
            return response.status_code, str(e)


def format_top_search_results(results: Dict[str, Any], top_n: int = 10) -> List[Dict[str, Any]]:
    """
    Format the top N results into a list of dictionaries with updated key names
    :param results:
    :param top_n:
    :return:
    """
    return [
        {
            "position": result.get("position"),
            "title": result.get('title'),
            "link": result.get('link'),
            "snippet": result.get('snippet')
        }
        for result in results.get("organic_results", [])[:top_n]
    ]

def search(search_query: str, location: str = "") -> str:
    """
    Main function to execute the Google search using SERP API and resturn top N results are a JSON string
    :param search_query:
    :param location:
    :return:
    """
    api_key = 'YOUR_KEY'
    serp_client = SerpAPIClient(api_key)
    results = serp_client(search_query, location=location)

    if isinstance(results, dict):
        top_results = format_top_search_results(results)
        return json.dumps({"top_results": top_results}, indent=2)
    else:
        status_code, error_message = results
        error_json = json.dumps({
            "error": f"Search failed with status code {status_code}: {error_message}"
        })
        return error_json



