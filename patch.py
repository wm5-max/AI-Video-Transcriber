import os
import sys

def patch():
    file_path = 'c:/Users/Mina/AI-Video-Transcriber/backend/summarizer.py'
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    old_text = 'strip_llm_artifacts(response.choices[0].message.content or "")'
    new_text = '(setattr(self, "_last_token_usage", self._get_token_usage(response)) or strip_llm_artifacts(response.choices[0].message.content or ""))'
    
    content = content.replace(old_text, new_text)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == "__main__":
    patch()
