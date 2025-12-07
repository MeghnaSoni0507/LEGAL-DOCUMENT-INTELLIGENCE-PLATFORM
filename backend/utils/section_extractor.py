import re

def extract_legal_sections(text):
    """
    Extract legal sections/clauses from document text.
    Looks for common patterns like "Article X", "Section Y", "Clause Z"
    """
    if not text:
        return []
    
    sections = []
    
    # Patterns to match section headers
    patterns = [
        r'(?:Article|ARTICLE)\s+([IVX\d]+)[:\.\s]+(.+)',
        r'(?:Section|SECTION)\s+(\d+(?:\.\d+)?)[:\.\s]+(.+)',
        r'(?:Clause|CLAUSE)\s+(\d+(?:\.\d+)?)[:\.\s]+(.+)',
        r'(\d+)\.\s+([A-Z][A-Za-z\s]{3,})',  # Numbered sections
    ]
    
    lines = text.split('\n')
    
    for i, line in enumerate(lines):
        for pattern in patterns:
            match = re.match(pattern, line.strip())
            if match:
                section_number = match.group(1)
                title = match.group(2).strip() if len(match.groups()) > 1 else "Untitled"
                
                # Get content (next few lines)
                content_lines = []
                for j in range(i + 1, min(i + 10, len(lines))):
                    if lines[j].strip():
                        content_lines.append(lines[j].strip())
                    else:
                        break
                
                sections.append({
                    'section_number': section_number,
                    'title': title[:100],  # Limit title length
                    'content': ' '.join(content_lines)[:500],  # Preview
                    'line_number': i + 1
                })
                break
    
    return sections