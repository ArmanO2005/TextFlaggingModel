import pandas as pd


def remove_non_utf8(text): 
    if isinstance(text, str): 
        return text.encode('utf-8', 'ignore').decode('utf-8', 'ignore') 
    return text 

data = pd.read_csv("C://Users//arman//0 - Chat AI Stuff//Suicide_Detection.csv")
data = data.applymap(remove_non_utf8)
only_suicide = data[(data['class'] == 'suicide') & (data['text'].str.len() < 100)]
suicide_dataset = data[data['class'].apply(lambda x : 'Bad')]
suicide_dataset.to_csv("C://Users//arman//0 - Chat AI Stuff//suicide_dataset.csv", index=False)