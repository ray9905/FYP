import os
import openai

gpt_key = os.getenv("OPENAI_KEY")



def llm_text_summary(eeg_result):
    gpt_prompt = f"The EEG classification result is {eeg_result} summarise this result in simple terms."
    return gpt_prompt
