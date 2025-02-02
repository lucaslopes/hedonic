import re
from collections import Counter, defaultdict
from pathlib import Path
import bibtexparser

def parse_bib_file(bib_path):
    with open(bib_path) as bibtex_file:
        bib_database = bibtexparser.load(bibtex_file)
    
    citations_info = {}
    for entry in bib_database.entries:
        key = entry.get('ID', '')
        title = entry.get('title', 'No title')
        author = entry.get('author', 'No author')
        year = entry.get('year', 'No year')
        citations_info[key] = {
            'title': title,
            'author': author,
            'year': year
        }
    return citations_info

def extract_paragraphs(tex_content):
    # Split content into paragraphs (separated by blank lines)
    paragraphs = re.split(r'\n\s*\n', tex_content)
    # Remove LaTeX comments and clean up whitespace
    paragraphs = [re.sub(r'%.*$', '', p, flags=re.MULTILINE).strip() for p in paragraphs]
    return [p for p in paragraphs if p]

def find_citations_in_paragraphs(paragraphs):
    citations_contexts = defaultdict(list)
    for para in paragraphs:
        # Find all citations in this paragraph
        citations = re.finditer(r'\\cite\{([^}]*)\}', para)
        for citation in citations:
            # Get the full citation string and its position
            cite_str = citation.group(1)
            cite_pos = citation.start()
            
            # Split multiple citations and clean
            refs = [ref.strip() for ref in cite_str.split(',')]
            for ref in refs:
                # Get context around the specific citation (100 chars before and after)
                start = max(0, cite_pos - 100)
                end = min(len(para), cite_pos + 200)
                context = '...' + ' '.join(para[start:end].split()) + '...'
                citations_contexts[ref].append(context)
    return citations_contexts

def count_citations(tex_path, bib_path, output_path):
    # Read the tex file
    tex_content = Path(tex_path).read_text()
    
    # Get bibliography information
    citations_info = parse_bib_file(bib_path)
    
    # Get paragraphs and find citations in context
    paragraphs = extract_paragraphs(tex_content)
    citations_contexts = find_citations_in_paragraphs(paragraphs)
    
    # Find all citations for counting
    citations = re.findall(r'\\cite\{([^}]*)\}', tex_content)
    
    # Split multiple citations
    all_refs = []
    for citation in citations:
        refs = [ref.strip() for ref in citation.split(',')]
        all_refs.extend(refs)
    
    # Count occurrences including all references from bib file with 0 counts if not cited
    citation_counts = Counter(all_refs)
    for ref in citations_info.keys():
        if ref not in citation_counts:
            citation_counts[ref] = 0
    
    # Prepare output content
    output_lines = [
        f"Citation counts for {tex_path}:",
        "-" * 80
    ]
    
    # Add citation counts sorted by frequency and then alphabetically
    for ref, count in sorted(citation_counts.items(), key=lambda x: (-x[1], x[0])):
        info = citations_info.get(ref, {'title': 'Not found in bib', 'author': 'N/A', 'year': 'N/A'})
        output_lines.append(
            f"{ref}: {count} time{'s' if count > 1 else ''}\n"
            f"    Title: {info['title']}\n"
            f"    Author(s): {info['author']}\n"
            f"    Year: {info['year']}"
        )
        
        # Only add contexts section if there are citations
        if count > 0:
            output_lines.append("    Appears in following contexts:")
            for i, context in enumerate(citations_contexts[ref], 1):
                output_lines.append(f"    {i}. {context}")
        
        output_lines.append("")  # Empty line between citations
    
    # Add summary
    output_lines.extend([
        "-" * 80,
        f"Total unique citations in bib: {len(citations_info)}",
        f"Total citations used in tex: {sum(c for c in citation_counts.values() if c > 0)}",
        f"Total uncited references: {sum(1 for c in citation_counts.values() if c == 0)}"
    ])
    
    # Write to file
    Path(output_path).write_text('\n'.join(output_lines))

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("Usage: python count_citations.py <path_to_tex_file> <path_to_bib_file> <output_file>")
        sys.exit(1)
    
    count_citations(sys.argv[1], sys.argv[2], sys.argv[3])
