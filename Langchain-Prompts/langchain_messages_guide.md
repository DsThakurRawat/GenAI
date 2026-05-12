# 💬 LangChain Prompting & Message Management Blueprint
*A robust reference manual mapping pure prompt templates against multi-role message arrays, automated contextual window pruning algorithms (`trim_messages`), and persistent session state wrappers.*

---

## 🏛️ 1. Complete Taxonomy of Core Message Roles

Unlike legacy dictionary schemas (`{"role": "user"}`), LangChain enforces explicit Pydantic-backed class boundaries. This normalizes arbitrary API layer parameters across disparate platform interfaces automatically.

```mermaid
graph TD
    classDef sys fill:#1e293b,stroke:#cbd5e1,stroke-width:1px,color:#fff;
    classDef hum fill:#0f172a,stroke:#38bdf8,stroke-width:2px,color:#fff;
    classDef ai fill:#022c22,stroke:#34d399,stroke-width:2px,color:#fff;

    Root["BaseMessage Protocol (Pydantic Base Class Interface)"]
    
    Root --> Sys["SystemMessage"] ::: sys
    Sys --> SysAttr["Defines persistent system constraints and persona traits."] ::: sys
    
    Root --> Hum["HumanMessage"] ::: hum
    Hum --> HumAttr["Carries arbitrary text/multimodal user query payloads."] ::: hum
    
    Root --> AI["AIMessage"] ::: ai
    AI --> AIAttr["Synthesized engine output containing tool_calls and token usage metadata."] ::: ai
```

### 📋 Role Implementation Reference Matrix
| Class Interface Type | Provider API Counterpart Role | Primary Injection Strategy | Lifetime Persistence Target |
| :--- | :--- | :--- | :--- |
| **`SystemMessage`** | `"system"` | Placed strictly at message array index `0`. | Static throughout session lifespan. |
| **`HumanMessage`** | `"user"` | Prepended sequentially as conversational tasks arrive. | Continually dynamic. |
| **`AIMessage`** | `"assistant"` | Returned synchronously by engine evaluation graphs. | Appended directly to chat history. |

---

## ✂️ 2. Advanced Context Window Optimization (`trim_messages`)

Unchecked message accumulation rapidly saturates model context window limits, inflating cloud processing costs and triggering attention mechanism diffusion.

LangChain provides the `.trim_messages()` utility to enforce rigorous, strategy-based sliding token truncation boundaries autonomously:

```mermaid
graph LR
    classDef keep fill:#022c22,stroke:#34d399,stroke-width:2px,color:#fff;
    classDef drop fill:#7f1d1d,stroke:#fca5a5,stroke-width:1px,color:#fff;

    subgraph Unpruned Message Array Stream
        M1["SysMsg"] ::: keep --> M2["Old_Hum"] ::: drop --> M3["Old_AI"] ::: drop --> M4["New_Hum"] ::: keep --> M5["New_AI"] ::: keep
    end
    
    subgraph strategy='last' Sliding Token Window Buffer
        M1 --> M4 --> M5
    end
```

### 💻 Production API Template Implementation:
```python
from langchain_core.messages import trim_messages

optimized_history = trim_messages(
    messages=raw_chat_history,
    max_tokens=4096,
    strategy="last",
    token_counter=chat_model_instance,
    include_system=True  # Strictly preserves initial SystemMessage integrity
)
```

---

## 💾 3. Persistent Session State Wrappers

Managing stateful chat histories across disconnected web requests requires dynamic routing wrappers linked directly to dedicated thread keys:

```mermaid
graph TD
    classDef core fill:#0f172a,stroke:#38bdf8,stroke-width:2px,color:#fff;
    classDef db fill:#312e81,stroke:#a5b4fc,stroke-width:2px,color:#fff;

    Req["Incoming API Task: session_id='user_99'"] --> Wrapper["RunnableWithMessageHistory"] ::: core
    Wrapper --> Fetch["Query Persistent Storage layer"] ::: db
    Fetch --> Inject["Inject cached array into current query thread"] ::: core
    Inject --> Model["Evaluate via LLM Engine"] ::: core
    Model --> Cache["Append generated AIMessage to storage"] ::: db
```

---

## 📁 4. Reference Scripts Map
Review executable code files directly inside this path to test template formatting capabilities:
- `prompt_template.py` / `chat_prompt_template.py`: Core foundational syntax setups.
- `messages.py` / `message_placeholder.py`: Array parameter serialization testing.
- `example_new_01_few_shot_prompt.py`: Injecting dynamic exemplary context blocks.
- `example_new_02_few_shot_chat_prompt.py`: Few-shot mapping across structured roles.
- `example_new_03_partial_prompts.py`: Pre-populating variables selectively.
- `example_new_04_pipeline_prompts.py`: Composing complex multi-prompt structures.
