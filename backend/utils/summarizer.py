def summarize_entire_document(text):
    """
    Generate basic document summaries.
    For AI-powered summaries, use the /ask-ai endpoint.
    """
    if not text:
        return ["No content to summarize"]
    
    word_count = len(text.split())
    char_count = len(text)
    
    # Extract first paragraph as preview
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    preview = paragraphs[0][:300] + "..." if paragraphs else "No preview available"
    
    return [
        f"Document contains approximately {word_count:,} words and {char_count:,} characters.",
        f"Preview: {preview}"
    ]


def summarize_section(section_text):
    """
    Summarize a specific section of text.
    """
    if not section_text:
        return "No content to summarize"
    
    # Return first 500 characters
    if len(section_text) <= 500:
        return section_text
    
    return section_text[:500] + "..."