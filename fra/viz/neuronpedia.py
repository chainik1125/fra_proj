"""
Neuronpedia URL and explanation utilities for FRA visualizations.
"""

import requests


def get_neuronpedia_url(layer: int, feature_idx: int, embed: bool = False) -> str:
    """
    Get Neuronpedia URL for a feature.

    Args:
        layer: Layer number
        feature_idx: Feature index
        embed: Whether to get embedded version

    Returns:
        Neuronpedia URL
    """
    base_url = f"https://www.neuronpedia.org/gpt2-small/{layer}-att-kk/{feature_idx}"
    if embed:
        return f"{base_url}?embed=true&embedexplanation=true&embedplots=true&embedtest=false"
    return base_url


def fetch_neuronpedia_explanation(layer: int, feature_idx: int, timeout: int = 2) -> str:
    """
    Fetch feature explanation from Neuronpedia API.

    Args:
        layer: Layer number
        feature_idx: Feature index
        timeout: Request timeout in seconds

    Returns:
        Feature explanation string or fallback message
    """
    import requests

    # Try to fetch from Neuronpedia API
    api_url = f"https://www.neuronpedia.org/api/feature/gpt2-small/{layer}-att-kk/{feature_idx}"

    try:
        response = requests.get(api_url, timeout=timeout)
        if response.status_code == 200:
            data = response.json()

            # Try to get explanation from various possible fields
            explanation = data.get('explanation', '')
            if not explanation:
                explanation = data.get('description', '')
            if not explanation:
                explanation = data.get('label', '')
            if not explanation and 'explanations' in data and data['explanations']:
                # Sometimes it's in an array
                explanation = data['explanations'][0].get('description', '')

            if explanation:
                return explanation

        return f"Feature activates on specific patterns (see Neuronpedia for details)"

    except Exception as e:
        # Fallback for network issues or API unavailable
        return f"Feature explanation unavailable (network error)"
