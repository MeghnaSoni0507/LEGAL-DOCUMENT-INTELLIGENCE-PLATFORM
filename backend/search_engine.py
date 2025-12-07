import re

def keyword_search(text, query):
    """
    Search for a keyword or phrase in the extracted text.
    Returns context snippets where matches occur.
    """
    if not text or not query:
        return ["No results found for your query."]
    
    # Escape special regex characters in query
    escaped_query = re.escape(query.strip())
    
    # Create pattern to find query with 60 chars context on each side
    pattern = re.compile(rf".{{0,60}}({escaped_query}).{{0,60}}", re.IGNORECASE)
    
    # Find all matches
    matches = list(pattern.finditer(text))
    
    if not matches:
        return ["No results found for your query."]
    
    results = []
    seen_snippets = set()  # Avoid duplicate snippets
    
    for match in matches:
        # Get context window
        start = max(0, match.start())
        end = min(len(text), match.end())
        
        # Extract snippet
        snippet = text[start:end].replace("\n", " ").strip()
        
        # Clean up multiple spaces
        snippet = re.sub(r'\s+', ' ', snippet)
        
        # Avoid duplicates
        if snippet not in seen_snippets and len(snippet) > 10:
            results.append(snippet)
            seen_snippets.add(snippet)
    
    return results if results else ["No results found for your query."]


def advanced_search(text, query, sections=None):
    """
    Enhanced search that returns structured results with section information.
    """
    # Get basic keyword matches
    snippets = keyword_search(text, query)
    
    # Count occurrences
    query_pattern = re.compile(re.escape(query), re.IGNORECASE)
    occurrences = len(query_pattern.findall(text))
    
    # Find which sections contain the query
    relevant_sections = []
    if sections:
        for section in sections:
            section_text = section.get('content', '')
            if query_pattern.search(section_text):
                relevant_sections.append({
                    'section_number': section.get('section_number', 'Unknown'),
                    'title': section.get('title', 'No title'),
                    'preview': section_text[:200] + '...' if len(section_text) > 200 else section_text
                })
    
    return {
        'query': query,
        'total_occurrences': occurrences,
        'snippets': snippets[:10],  # Limit to top 10 snippets
        'relevant_sections': relevant_sections[:5],  # Top 5 sections
        'found': occurrences > 0,
        'results': snippets[:10]  # For backward compatibility
    }