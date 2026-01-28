# FRA RAG System - Frontend Package

## Overview
Arabic RAG (Retrieval-Augmented Generation) system UI for the Egyptian Financial Regulatory Authority (FRA).

## Files
```
frontend_package/
├── index.html      # Main HTML structure
├── styles.css      # All styling (RTL Arabic support, dark mode)
├── app.js          # Frontend logic & API calls
└── README_FOR_FRONTEND_ENGINEER.md
```

## Tech Stack
- **Pure HTML5 / CSS3 / JavaScript (ES6+)**
- **No frameworks required** (vanilla JS)
- **Font:** Cairo (Google Fonts)
- **RTL Support:** Built-in

## Current UI Components

### Main Chat Interface
- Chat input box (Arabic RTL)
- Response display with markdown
- Evidence/Sources expandable section

### Sidebar Controls
| Component | Type | Purpose |
|-----------|------|---------|
| Language Dropdown | Dropdown | العربية / English |
| Number of Sources | Slider | 1-10 sources |
| Hybrid Search | Checkbox | Enable/disable |
| Reranking | Checkbox | Enable/disable |
| ReAct Agent | Checkbox | Multi-hop reasoning |
| Entity Filter | Dropdown | Filter by entity type |
| Doc Type Filter | Dropdown | Filter by document type |
| Topic Filter | Dropdown | Filter by topic |
| File Upload | File | Upload documents |
| Statistics Button | Button | Show DB stats |
| Browse Chunks Button | Button | View stored chunks |

### Feedback System
- Thumbs up/down buttons
- Feedback saved to JSON

## Key Functions to Modify

### UI Layout (line ~600-900)
```python
with gr.Blocks(...) as demo:
    # Main layout here
```

### Chat Handler (line ~130-230)
```python
def chat_with_rag(message, history, language, ...):
    # Handles user messages
```

### Evidence Display (line ~318-369)
```python
def _build_evidence_text(sources, language, query):
    # Builds the sources/evidence HTML
```

## Integration Points

When returning the enhanced frontend, ensure these function signatures remain unchanged:

```python
# Main chat function - 
def chat_with_rag(
    message: str,
    history: List,
    language: str,
    num_sources: int,
    use_hybrid: bool,
    use_rerank: bool,
    use_react: bool,
    entity_filter: str,
    doc_type_filter: str,
    topic_filter: str,
) -> Tuple[List, str, str, str, str, str]:
    ...

# Statistics function
def get_stats() -> str:
    ...

# Browse chunks function  
def browse_chunks(limit: int = 10) -> str:
    ...

# Upload function
def upload_and_index_documents(files) -> str:
    ...
```

## Styling Notes
- RTL support for Arabic: `direction: rtl; text-align: right;`
- Current theme: `gr.themes.Soft()`
- Primary color: Blue gradient (#007bff)

## Running Locally
Just open `index.html` in a browser - no server required for UI development.

For testing with live reload:
```bash
npx live-server
```

## API Integration Points
In `app.js`, replace the stub functions with actual API calls:

| Function | Endpoint | Purpose |
|----------|----------|---------|
| `sendToBackend()` | POST /api/chat | Send user message, get answer |
| `uploadToBackend()` | POST /api/upload | Upload documents |
| `getStatsFromBackend()` | GET /api/stats | Get system statistics |
| `getChunksFromBackend()` | GET /api/chunks | Browse stored chunks |

## Features Included
- Dark/Light theme toggle
- RTL Arabic layout
- Responsive design
- Typing indicator
- Sources panel
- Feedback buttons ()
- File upload with drag & drop
- Settings persistence (localStorage)

## Requested Enhancements
1. Modern, cleaner UI design
2. Better mobile responsiveness
3. Improved evidence/sources display
4. Loading animations
5. Better error states

## Return
Return the enhanced HTML/CSS/JS files for integration with the Python backend.
