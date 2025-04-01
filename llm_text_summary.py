import os
import openai

client = openai.OpenAI(api_key= os.getenv("OPENAI_API_KEY"))


def llm_text_summary(eeg_result):
    gpt_prompt = f"The EEG classification result is {eeg_result} summarise this result in simple terms."
    gpt_response = client.chat.completions.create(
        model = "gpt-3.5-turbo",
        messages = [ 
            {"role":"system", "content": "You are an expert neurologist summarising EEG results."},
            {"role":"user", "content": gpt_prompt}
        ]
    )

    return gpt_response.choices[0].message.content

print(llm_text_summary("abnormal"))   

 
   
    
    
                                     

