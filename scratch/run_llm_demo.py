import sys, os, asyncio
# Ensure project root is on import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from llm_service import generate_answer

# Provide a minimal chunk list – the context builder expects a list of dicts with a "content" key
chunks = [{"content": "Test context for LLM."}]

async def main():
    async for token in generate_answer("What is this?", chunks, "demo-id"):
        print(token, end='')

if __name__ == "__main__":
    asyncio.run(main())
