from typing import Optional, Union, Tuple, Dict, Any, List
import json
import wikipediaapi


def search(query: str) -> Optional[str]:
    """
    Fetch Wikipedia information for a given search query using wikipedia-API and return a JSON
    :param query: The search query string
    :return: Optional[str]: A JSON string containing the query, title and summary, or None if no
    result is found
    """
    wiki = wikipediaapi.Wikipedia("ReAct Learning (abhishekmazumdar94@gmail.com)", "en")

    try:
        page = wiki.page(query)

        if page.exists():

            result = {
                "query": query,
                "title": page.title,
                "summary": page.summary
            }

            return json.dumps(result, ensure_ascii=False, indent=2)
        else:
            return None
    except Exception as e:
        print(f"Error at Wiki Tool:: {e}")
        return None

