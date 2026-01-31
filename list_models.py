import google.generativeai as genai

genai.configure(api_key="AIzaSyBIUNbznInrzSrdEw7tFcPn9MuAuyj_9Hc")
for m in genai.list_models():
    print(m.name, m.supported_generation_methods)
