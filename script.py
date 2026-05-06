import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import os
import json

load_dotenv()

api_key = os.getenv('GROQ_API_KEY')

client = OpenAI(api_key=api_key, base_url='https://api.groq.com/openai/v1')

file_path = 'data.csv'

def load_reviews():
    reviews = pd.read_csv(file_path)
    review_column = reviews.columns[0]
    return reviews[review_column].to_list()

def analysis_llm(data):
    prompt = f'''
    Проанализируй тональность следующих отзывов клиентов.
    Для каждого отзыва определи: positive, negative или neutral.
    {data}
    Верни только JSON без любого другого текста:
    {{
        'total_reviews': {len(data)},
        'mood_distribution': {{
            'positive': int,
            'negative': int,
            'neutral': int
        }},
        'results': [
            {{
                'review_id': int,
                'mood': 'positive,negative или neutral',
                'reason': 'Краткое объяснение'
            }}
        ],
        'summary': 'Общий вывод по настроению отзывов'
    }}
    '''

    response = client.chat.completions.create(
        model='llama-3.3-70b-versatile',
        messages=[
            {'role': 'user', 'content': prompt}
        ],
        temperature=0.2,
        max_tokens=4000
    )
    content = response.choices[0].message.content.strip()
    if '```' in content:
        content = content.split('```')[1].strip()
    return json.loads(content)
    

reviews = load_reviews()
result = analysis_llm(reviews)
    
with open("result.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
