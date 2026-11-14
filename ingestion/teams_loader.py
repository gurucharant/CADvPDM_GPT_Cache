from pathlib import Path
import pandas as pd
PROJECT_ROOT=Path(__file__).resolve().parents[1]
FILE=PROJECT_ROOT/'data/teams/teams_transcripts.csv'
def load_teams_documents():
    df=pd.read_csv(FILE); docs=[]
    for _,r in df.iterrows():
        content=f"""Teams Meeting Knowledge
Meeting ID: {r.meeting_id}
Title: {r.title}
Application: {r.application}
Site: {r.site}
Error Code: {r.error_code}
Summary: {r.summary}
Decisions: {r.decisions}
Action Items: {r.action_items}"""
        docs.append({'source_type':'TEAMS','source_id':str(r.meeting_id),'chunk_index':0,'title':str(r.title),'content':content,'site':str(r.site),'application':str(r.application),'error_code':str(r.error_code),'metadata':{'meeting_ts':str(r.meeting_ts),'organizer':str(r.organizer),'participants':str(r.participants)}})
    return docs
