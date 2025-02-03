import os
import openai

client = openai.OpenAI(api_key= os.getenv("OPENAI_API_KEY"))


def llm_text_summary(eeg_result):
    gpt_prompt = f"The EEG classification result is {eeg_result} summarise this result in simple terms."
    
    
                                     

