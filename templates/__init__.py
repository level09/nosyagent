from pathlib import Path
import markdown
from datetime import datetime
from jinja2 import Environment, FileSystemLoader

# Initialize Jinja2 environment
template_dir = Path(__file__).parent
jinja_env = Environment(loader=FileSystemLoader(template_dir))

def render_document_template(title, content):
    """Render markdown content using the document template"""
    template = jinja_env.get_template("document.html")
    
    # Convert markdown to HTML
    html_content = markdown.markdown(content)
    
    return template.render(
        title=title,
        content=html_content,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )