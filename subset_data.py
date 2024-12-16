import pandas as pd
import json
import spacy
import string
df = pd.read_csv('/data/HowTo100M_v1.csv')
sliced_df = df.sample(n=2000, random_state=42)
video_ids = df['video_id']
video_path = 'http://howto100m.inria.fr/dataset/' + video_ids
video_path = video_path.tolist()

with open('howto100m_videos.txt', 'w') as f:
    for url in video_path:
        f.write(url + '\n')

video_caption_sliced = {}
file_path = '/data/caption.json'

with open(file_path, "r") as file:
    data = json.load(file)

for url in video_path:
    url=url.split('/')[-1]
    video_caption_sliced[url] = data[url]


with open("video_caption_sliced.json", "w") as f:
    json.dump(video_caption_sliced, f, indent=4)


def preprocess(words):

    nlp = spacy.load("en_core_web_sm")

    text = " ".join(words)
    
    doc = nlp(text)
    
    cleaned_words = []
    for token in doc:
        if not token.is_stop and not token.is_punct:
            cleaned_words.append(token.lemma_.lower())  
    return cleaned_words

