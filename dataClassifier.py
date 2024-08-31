import pandas as pd
from openai import OpenAI

client = OpenAI(
    base_url = 'http://localhost:11434/v1',
    api_key='ollama', # required, but unused
)

uncleanedData = pd.read_csv("C://Users//arman//0 - Chat AI Stuff//untrained_data_test.csv")

# example = pd.DataFrame(
#     {'Title': [
#     'I love my dog',
#     'Im gonna shoot up the school',
#     'im gonna shoot some hoops',
#     'I want to buy cocaine'],
#     'Classification': ['', '', '', '']})

def classify_document(document):
    response = client.chat.completions.create(
        model="llama3.1",
        messages=[  
            {"role": "system", "content": "You are a message classifier. You receive SMS data and classify it, you do not answer questions or engage in conversation. If a message includes mention of illegal drug use, pornography, guns, or other not safe for work content, return 'Bad'. Else, return 'Good'. Assume messages are good until there is a clear reason why they would be bad. Return either the word 'Good' or the word 'Bad'."},
            {"role": "user", "content": document}
        ]
    )

    print(response.choices[0].message.content)

    if ('Good' or 'good') in response.choices[0].message.content:
        return 'Good'
    elif ('Bad' or 'bad' or 'illicit substance' or 'illegal' ) in response.choices[0].message.content:
        return 'Bad'
    else:
        return 'unsure'


def classify_df(csv):
    for i, document in enumerate(csv['Title']):
        print(i)
        csv.loc[i, 'Classification'] = classify_document(csv.loc[i, 'Title'])


def See(output_csv):
    #dont run this for the actual file or python will die :(
    for i in range(len(output_csv)):
        print(output_csv.loc[i, 'Classification'])

classify_df(uncleanedData)

uncleanedData.to_csv('C://Users//arman//0 - Chat AI Stuff//classified_data.csv', index=False)
