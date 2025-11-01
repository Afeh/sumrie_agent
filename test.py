# from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

# ytt_api = YouTubeTranscriptApi()
# # print(ytt_api.fetch("j31dmodZ-5c"))

# transcript_list = ytt_api.fetch("j31dmodZ-5c")
# transcript_text = " ".join([snippet.text for snippet in transcript_list.snippets])
# print(transcript_text)


import requests
import json

response = requests.post(
  url="https://openrouter.ai/api/v1/chat/completions",
  headers={
    "Authorization": "Bearer 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",
  },
  data=json.dumps({
    "model": "google/gemma-3n-e2b-it:free", # Optional
    "messages": [
      {
        "role": "user",
        "content": "What is the meaning of life?"
      }
    ]
  })
)

print(response.json())