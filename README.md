# 🧠 Generative AI Notes

Beginner-friendly notes on the core building blocks of GenAI apps, with animated diagrams.

## 📑 Contents
1. [Tokens](#1--tokens)
2. [Zero-shot and Few-shot Prompting](#2--zero-shot-and-few-shot-prompting)
3. [API](#3--api)
4. [MCP (Model Context Protocol)](#4--mcp-model-context-protocol)
5. [RAG (Retrieval-Augmented Generation)](#5--rag-retrieval-augmented-generation)
6. [Cheat Sheet](#-cheat-sheet)

---

## 1. 🔤 Tokens

![Tokens animation](assets/tokens.svg)

**Plain language:** an LLM does not read letters or words. It reads **tokens**, small chunks of text (a word, part of a word, or punctuation). Each token is mapped to a number (an ID), and the model works on those numbers.

| Fact | Detail |
|---|---|
| Rough size | 1 token is about 4 characters of English, or about 0.75 words |
| Long or rare words | Split into several tokens (`building` → `build` + `ing`) |
| Non-English text and code | Often use more tokens per word |
| **Context window** | The maximum tokens the model can handle at once (prompt + reply) |
| **Cost and speed** | Billed per input token and per output token |
| `max_tokens` | Upper limit on how many tokens the model may generate |

**Why it matters:** shorter prompts are cheaper and faster, and anything beyond the context window is lost or rejected. This is why RAG sends only the relevant chunks instead of whole documents.

---

## 2. 🎯 Zero-shot and Few-shot Prompting

![Prompting animation](assets/prompting.svg)

**Prompt** = the input you give the model. **Prompt engineering** = writing it so you get reliable output.

| | Zero-shot | Few-shot |
|---|---|---|
| Examples in prompt | 0 | 2 to 5 (more is possible) |
| Relies on | Model's pre-trained knowledge | Pattern shown in your examples |
| Best for | Simple, common tasks | Custom formats, tone, labels, tricky edge cases |
| Cost | Fewer tokens | More tokens (examples count) |

### Zero-shot example
```text
Classify the sentiment as Positive or Negative:
"The battery dies so fast."
```

### Few-shot example
```text
Classify the sentiment as Positive or Negative.

Review: "Loved it!"            -> Positive
Review: "Waste of money"       -> Negative
Review: "Works great"          -> Positive
Review: "The battery dies so fast." ->
```

### Tips
- Start with zero-shot. Add examples only if the output is wrong or inconsistent.
- Keep example **format identical**, because the model copies it.
- Use varied examples, including at least one edge case.
- Be specific: role, task, format, length (`Answer in 3 bullet points`).
- Other techniques: **Chain-of-thought** ("think step by step"), **system prompts**, **structured output** (ask for JSON).

---

## 3. 🔌 API

![API animation](assets/api.svg)

**Plain language:** an API is a doorway. Your code sends an HTTP request containing the prompt, and the provider's server runs the model and sends back a response.

**Key parts of a request**

| Field | Meaning |
|---|---|
| `model` | Which model to use |
| `messages` | The conversation (roles: `user`, `assistant`) |
| `system` | Instructions that set behaviour |
| `max_tokens` | Max length of the reply |
| `temperature` | Randomness (low = focused, high = creative) |
| `stream` | Send tokens as they are generated |

### Python example (Anthropic API)
```bash
pip install anthropic
export ANTHROPIC_API_KEY="your-key-here"   # keep this OUT of GitHub
```
```python
import anthropic

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from environment

msg = client.messages.create(
    model="claude-sonnet-5-5",
    max_tokens=300,
    system="You are a helpful tutor.",
    messages=[{"role": "user", "content": "Explain tokens in 2 sentences."}],
)

print(msg.content[0].text)
print(msg.usage.input_tokens, msg.usage.output_tokens)  # token usage
```

**Good practice:** store keys in environment variables or `.env` (add `.env` to `.gitignore`), handle rate limit errors with retries, and track token usage.

---

## 4. 🔗 MCP (Model Context Protocol)

![MCP animation](assets/mcp.svg)

**Plain language:** MCP is like **USB-C for AI**. Instead of writing a custom integration for every tool, any MCP-capable AI app can connect to any MCP server in the same standard way.

| Role | What it is |
|---|---|
| **Host** | The AI app (e.g. Claude Desktop, an IDE) |
| **Client** | Lives inside the host, keeps one connection per server |
| **Server** | A small program exposing capabilities (GitHub, Drive, database...) |

**What a server can expose**
- 🛠 **Tools**: actions the model can call (`create_issue`, `run_query`)
- 📄 **Resources**: data to read (files, records)
- 💬 **Prompts**: reusable prompt templates

**How it works:** messages use JSON-RPC 2.0 over a transport, either `stdio` (local process) or HTTP. Flow: connect, discover tools, the model picks a tool, the client calls it, the result goes back to the model.

### Minimal MCP server (Python)
```bash
pip install "mcp[cli]"
```
```python
# server.py
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("demo")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

if __name__ == "__main__":
    mcp.run()   # runs over stdio by default
```

**API vs MCP:** an API is how *you* call one specific service. MCP is a common standard for how *an AI* discovers and uses many tools.

---

## 5. 📚 RAG (Retrieval-Augmented Generation)

![RAG animation](assets/rag.svg)

**Plain language:** the model only knows its training data. RAG lets it **look things up in your documents first**, then answer using what it found.

### Pipeline
**Indexing (once)**
1. **Load** documents (PDF, web, notes)
2. **Chunk** into pieces (about 300 to 1000 tokens, with small overlap)
3. **Embed** each chunk into a vector (a list of numbers capturing meaning)
4. **Store** vectors in a vector database (FAISS, Chroma, Pinecone...)

**Querying (every question)**
1. **Embed** the question
2. **Retrieve** the top-k most similar chunks (cosine similarity)
3. **Augment**: put those chunks into the prompt
4. **Generate**: the LLM answers from the context, ideally citing sources

### Tiny working example
```bash
pip install sentence-transformers numpy
```
```python
import numpy as np
from sentence_transformers import SentenceTransformer

docs = [
    "RAG retrieves relevant text before generating an answer.",
    "Tokens are small pieces of text the model reads.",
    "MCP standardises how AI apps connect to tools.",
]

model = SentenceTransformer("all-MiniLM-L6-v2")
doc_vecs = model.encode(docs, normalize_embeddings=True)

query = "What does MCP do?"
q_vec = model.encode([query], normalize_embeddings=True)[0]

scores = doc_vecs @ q_vec                 # cosine similarity
top = np.argsort(scores)[::-1][:2]        # top-2 chunks
context = "\n".join(docs[i] for i in top)

prompt = f"Answer using only this context:\n{context}\n\nQuestion: {query}"
print(prompt)   # send this prompt to any LLM API
```

### RAG vs Fine-tuning
| | RAG | Fine-tuning |
|---|---|---|
| Adds | Fresh or private knowledge | New style or behaviour |
| Update data | Easy (re-index) | Retrain |
| Cost | Lower | Higher |
| Sources and citations | Yes | No |

### Common problems
- Bad chunking leads to missing context, so tune chunk size and overlap.
- Retrieval misses, so try hybrid search (keyword + vector) or a reranker.
- Too many chunks waste tokens and confuse the model, so keep top-k small.

---

## ✅ Cheat Sheet

| Concept | One line |
|---|---|
| Token | Chunk of text the model reads; cost and limits are counted in tokens |
| Zero-shot | Just the instruction, no examples |
| Few-shot | Instruction plus a few examples |
| API | HTTP doorway to call the model from code |
| MCP | Standard plug connecting AI apps to tools and data |
| RAG | Retrieve relevant docs, then generate a grounded answer |

---
⭐ If these notes helped, star the repo!
