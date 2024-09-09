import pandas as pd
from openai import OpenAI

client = OpenAI(
    base_url = 'http://localhost:11434/v1',
    api_key='ollama', # required, but unused
)

uncleanedData = pd.read_csv("C://Users//arman//0 - Chat AI Stuff//unclassified_data_2.csv")

badStuff = ['Dangerous', 'dangerous', 'illicit', 'illegal', 'hate', 'criminal', 'substance', 'explicit', 'malicious', 'drugs']
goodStuff = ['Safe', 'safe']

def classify_document(document):
    response = client.chat.completions.create(
        model="llama3.1",
        messages=[  
            {"role": "system", "content": "You are a message classifier. You receive SMS data and classify it, you do not answer questions or engage in conversation. If a message includes mention of illegal drug use, sex, guns, alcohol or explicit or hateful language, return 'Dangerous'. Else, return 'Safe'. Assume messages are safe until there is a clear reason why they would be dangerous. Return either the word 'Safe' or the word 'Dangerous'."},
            {"role": "user", "content": document}
        ]
    )  

    response = response.choices[0].message.content.lower()
    print(response)

    if any([word in response for word in goodStuff]):
        return 'Good'
    elif any([word in response for word in badStuff]):
        print(document)
        return 'Bad'
    else:
        print(document)
        return 'Unsure'


def classify_df(csv):
    for i, document in enumerate(csv['Title']):
        print(i)
        csv.loc[i, 'Classification'] = classify_document(csv.loc[i, 'Title'])


def See(output_csv):
    #dont run this for the actual file or python will die :(
    for i in range(len(output_csv)):
        print(output_csv.loc[i, 'Classification'])

classify_df(uncleanedData)

uncleanedData.to_csv('C://Users//arman//0 - Chat AI Stuff//classified_data_2.csv', index=False)
