# 🌐 Enterprise Project Spec: Universal Context Browser Assistant
*An end-to-end design blueprint mapping a multi-browser Chrome/Brave extension frontend against an asynchronous LangChain backend microservice to summarize dynamic web pages and transcribe YouTube videos directly in-context.*

---

## 🏛️ 1. High-Level System Architecture

The solution decouples the stateless client browser extension from a centralized serverless Python execution layer responsible for orchestrating document parsing, text chunking, and language model inference graphs.

```mermaid
graph TD
    classDef client fill:#0f172a,stroke:#38bdf8,stroke-width:2px,color:#fff;
    classDef proxy fill:#1e293b,stroke:#cbd5e1,stroke-width:1px,color:#fff;
    classDef backend fill:#312e81,stroke:#a5b4fc,stroke-width:2px,color:#fff;

    subgraph Client Browser Extension Layer
        Popup["Extension UI: Popup Window Action"] ::: client
        Script["Injected ContentScript: Extract Active URI"] ::: client
    end
    
    subgraph REST/WebSocket Transport Proxy
        Gateway["Cloud API Gateway API / Auth Filter"] ::: proxy
    end
    
    subgraph LangChain Backend Service (FastAPI Server)
        Router{"Inspect Target URI Protocol"} ::: backend
        WebRoute["Standard URL: WebBaseLoader"] ::: backend
        YTRoute["YouTube URL: YoutubeLoader"] ::: backend
        
        Split["RecursiveCharacterTextSplitter / Token Tracker"] ::: backend
        Model["Inference Model Pipeline (Gemini/GPT-4o)"] ::: backend
    end

    Popup --> Script
    Script -- HTTP POST Payload --> Gateway
    Gateway --> Router
    Router -- Plain Webpage --> WebRoute
    Router -- Video Page --> YTRoute
    WebRoute --> Split
    YTRoute --> Split
    Split --> Model
    Model -- Markdown Stream Summary --> Popup
```

---

## ⚙️ 2. Core Operational Modules

### 🖥️ 1. Extension Manifest (v3) Frontend
- **Permissions**: `activeTab`, `scripting`.
- **User Experience**: Triggering the extension displays a polished glassmorphism interface floating over the current tab. Renders received markdown summaries in real-time.

### 🐍 2. Python Backend Services Architecture
The processing layer relies on two specialized LangChain document extractors:

```mermaid
graph LR
    classDef default fill:#1e293b,stroke:#cbd5e1,stroke-width:1px,color:#fff;
    classDef target fill:#022c22,stroke:#34d399,stroke-width:2px,color:#fff;

    Root["Incoming Target Link"]
    
    Root --> TextLink["Arbitrary Web Article URI"]
    TextLink --> WebBase["langchain_community.document_loaders.WebBaseLoader"] ::: target
    WebBase --> TextExtraction["Strips raw HTML tags;<br/>Yields minified text string array."]
    
    Root --> VideoLink["youtube.com/watch?v=XXXX"]
    VideoLink --> YTLoader["langchain_community.document_loaders.YoutubeLoader"] ::: target
    YTLoader --> Subtitles["Downloads native caption streams automatically;<br/>Bypasses pixel processing overhead entirely."]
```

---

## 📐 3. Implementation Blueprint: Execution Phasing

### Phase 1: Prototype Engine Validation
- Build local standalone FastAPI script encapsulating conditional routing logic mapping links to target document loaders.
- Test offline output parsers to ensure robust markdown headers extraction.

### Phase 2: Client Extension Interface Assembly
- Build a lightweight Vanilla JS manifest configuration calling local Python backend targets securely.
- Enforce strict Content Security Policy (CSP) headers to validate cross-origin requests.

### Phase 3: Token Budgeting & Streaming Integration
- Attach token cost tracking callbacks directly to execution streams.
- Shift return transports from blocking static strings to chunked Server-Sent Events (SSE) interfaces to deliver real-time visual progress loops directly to client user popups.
