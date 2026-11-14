from pathlib import Path
import pandas as pd
PROJECT_ROOT=Path(__file__).resolve().parents[1]
FILE=PROJECT_ROOT/'data/git/git_activity.csv'
def load_git_documents():
    df=pd.read_csv(FILE); docs=[]
    for _,r in df.iterrows():
        content=f"""Git Engineering Activity
Activity ID: {r.activity_id}
Type: {r.activity_type}
Repository: {r.repository}
Application: {r.application}
Site: {r.site}
Related Error: {r.related_error_code}
Summary: {r.summary}
Details: {r.details}"""
        docs.append({'source_type':'GIT','source_id':str(r.activity_id),'chunk_index':0,'title':str(r.summary),'content':content,'site':str(r.site),'application':str(r.application),'error_code':str(r.related_error_code),'metadata':{'timestamp':str(r.timestamp),'repository':str(r.repository),'activity_type':str(r.activity_type),'author':str(r.author)}})
    return docs
