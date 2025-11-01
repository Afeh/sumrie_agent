import os
import google.generativeai as genai
import httpx # Import the new HTTP client
import re
from uuid import uuid4
from typing import Optional
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from models.a2a import A2AMessage, TaskResult, TaskStatus, Artifact, MessagePart



class YouTubeSummarizerAgent:
    def __init__(self, provider: str, google_api_key: str, openrouter_api_key: str, openrouter_model: str):

        self.provider = provider
        self.http_client = httpx.AsyncClient()
        if self.provider == "google":
            if not google_api_key:
                raise ValueError("GOOGLE_API_KEY is required for the 'google' provider.")
            genai.configure(api_key=google_api_key)
            self.summarization_model = genai.GenerativeModel('models/gemini-2.5-pro')#gemma-3n-e2b-it:free
            print("YouTube Summarizer configured to use: Google Gemini")

        elif self.provider == "openrouter":
            if not openrouter_api_key:
                raise ValueError("OPENROUTER_API_KEY is required for the 'openrouter' provider.")
            self.openrouter_api_key = openrouter_api_key
            self.openrouter_model = openrouter_model
            print(f"YouTube Summarizer configured to use: OpenRouter (Model: {self.openrouter_model})")
            
        else:
            raise ValueError(f"Unknown LLM_PROVIDER: '{self.provider}'. Must be 'google' or 'openrouter'.")

    def _get_video_id(self, url: str) -> Optional[str]:
        """Extracts the YouTube video ID from a URL."""
        if not url:
            return None
        
        query = urlparse(url)
        if query.hostname == 'youtu.be':
            return query.path[1:]
        if query.hostname in ('www.youtube.com', 'youtube.com'):
            if query.path == '/watch':
                p = parse_qs(query.query)
                return p.get('v', [None])[0]
            if query.path[:7] == '/embed/':
                return query.path.split('/')[2]
            if query.path[:3] == '/v/':
                return query.path.split('/')[2]
        
        return None

    def _get_transcript_from_api(self, video_id: str) -> str:
        """Fetches a transcript and formats it into a single string."""
        try:
            # This fetches the transcript data
            transcript_list = YouTubeTranscriptApi().fetch(video_id)
            transcript_text = " ".join([snippet.text for snippet in transcript_list.snippets])
            return transcript_text

        except (TranscriptsDisabled, NoTranscriptFound) as e:
            # This is our graceful error handling
            print(f"Could not retrieve transcript for video {video_id}: {e}")
            raise ValueError(f"A transcript is not available for this video. Captions may be disabled.")
        except Exception as e:
            print(f"An unexpected error occurred while fetching transcript: {e}")
            raise ValueError("An unexpected error occurred while trying to get the video transcript.")


    async def _summarize_text(self, transcript: str) -> str:
        """Summarizes the transcript using the configured provider."""
        prompt = f"Please summarize the following transcript:\n\n{transcript}"
        system_prompt = "You are an expert at summarizing YouTube video transcripts. Provide a concise, easy-to-read summary that captures the key points. Crucially, the entire response must be in plain text, with no Markdown formatting (no headers, bold text, or lists)."

        if self.provider == "google":
            full_prompt = f"{system_prompt}\n\n{prompt}"
            response = self.summarization_model.generate_content(full_prompt)
            return response.text

        elif self.provider == "openrouter":
            response = await self.http_client.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={ "Authorization": f"Bearer {self.openrouter_api_key}" },
                json={
                    "model": self.openrouter_model,
                    "messages": [
                        { "role": "system", "content": system_prompt },
                        { "role": "user", "content": prompt }
                    ]
                }
            )
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']

    async def process_message(self, message: A2AMessage) -> TaskResult:
        task_id = message.taskId or str(uuid4())
        context_id = str(uuid4())

        user_text = next((part.text for part in message.parts if part.kind == "text"), None)

        # FIX #2: Extract the URL from the user's text using regex.
        if not user_text:
            raise ValueError("No text found in the message.")

        # This pattern finds most standard YouTube URLs
        url_pattern = re.compile(r'(https?://(www\.)?(youtube\.com/watch\?v=|youtu\.be/)[\w-]+)')
        match = url_pattern.search(user_text)

        if not match:
            # If the regex didn't find a URL at all
            raise ValueError("No valid YouTube URL found in the message.")

        # Pass the CLEAN, extracted URL to the parsing function
        extracted_url = match.group(0)
        video_id = self._get_video_id(extracted_url)
        
        # This check is now a fallback, in case the URL is malformed in a weird way
        if not video_id:
            raise ValueError("Could not extract a valid video ID from the URL.")

        try:
            # 2. Get Transcript via API
            print(f"[{task_id}] Fetching transcript for video ID: {video_id}...")
            transcript = self._get_transcript_from_api(video_id)

            # 3. Summarize Transcript
            print(f"[{task_id}] Summarizing transcript...")
            summary = await self._summarize_text(transcript)

            # 4. Build a successful response
            response_message = A2AMessage(
                role="agent",
                parts=[MessagePart(kind="text", text=summary)],
                taskId=task_id
            )
            artifacts = [
                Artifact(name="summary", parts=[MessagePart(kind="text", text=summary)]),
                Artifact(name="full_transcript", parts=[MessagePart(kind="text", text=transcript)])
            ]
            return TaskResult(
                id=task_id,
                contextId=context_id,
                status=TaskStatus(state="completed", message=response_message),
                artifacts=artifacts,
                history=[message, response_message]
            )
        except ValueError as e:
            # 5. Handle cases where a transcript isn't found or another error occurs
            error_text = str(e)
            error_message = A2AMessage(
                role="agent",
                parts=[MessagePart(kind="text", text=error_text)],
                taskId=task_id
            )
            return TaskResult(
                id=task_id,
                contextId=context_id,
                status=TaskStatus(state="failed", message=error_message),
                history=[message, error_message]
            )