import os
import google.generativeai as genai
import httpx
import re
from uuid import uuid4
from typing import Optional
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from models.a2a import A2AMessage, TaskResult, TaskStatus, Artifact, MessagePart, MessageConfiguration


class YouTubeSummarizerAgent:
    def __init__(self, provider: str, google_api_key: str, openrouter_api_key: str, openrouter_model: str):
        self.provider = provider
        self.http_client = httpx.AsyncClient()
        if self.provider == "google":
            if not google_api_key: raise ValueError("GOOGLE_API_KEY is required for the 'google' provider.")
            genai.configure(api_key=google_api_key)
            self.summarization_model = genai.GenerativeModel('gemini-1.5-pro-latest')
            print("YouTube Summarizer configured to use: Google Gemini")
        elif self.provider == "openrouter":
            if not openrouter_api_key: raise ValueError("OPENROUTER_API_KEY is required for the 'openrouter' provider.")
            self.openrouter_api_key = openrouter_api_key
            self.openrouter_model = openrouter_model
            print(f"YouTube Summarizer configured to use: OpenRouter (Model: {self.openrouter_model})")
        else:
            raise ValueError(f"Unknown LLM_PROVIDER: '{self.provider}'. Must be 'google' or 'openrouter'.")

    # --- NEW METHOD TO SEND WEBHOOKS ---
    async def _send_webhook_notification(self, url: str, token: str, result: TaskResult):
        """Sends the final task result to the provided webhook URL."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }
        try:
            # THE FIX: Check if there is a message to send, and send ONLY the message.
            if result.status and result.status.message:
                payload = result.status.message.model_dump(exclude_none=True)
                print(f"Sending final MESSAGE to webhook: {url}")
                response = await self.http_client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                print("Webhook notification sent successfully.")
            else:
                print("Webhook notification skipped: No final message in the task result.")

        except httpx.HTTPError as e:
            print(f"Error sending webhook notification: {e}")
            # Optional: Log the response body for more details on the error
            if 'response' in locals():
                print(f"Webhook error response body: {response.text}")


    # This is the background task that does the actual work
    async def _do_summarization_and_notify(self, message: A2AMessage, webhook_url: str, webhook_token: str):
        """Runs the summarization and sends the result to a webhook."""
        # This re-uses the same logic as the blocking mode
        final_result = await self.process_message(message, MessageConfiguration(blocking=True))
        await self._send_webhook_notification(webhook_url, webhook_token, final_result)

    # --- UPDATED PROCESS_MESSAGE TO HANDLE BOTH MODES ---
    async def process_message(self, message: A2AMessage, config: MessageConfiguration, background_tasks=None) -> TaskResult:
        """The main processing pipeline for a summarization task."""
        
        # --- NON-BLOCKING (WEBHOOK) LOGIC ---
        if not config.blocking and config.pushNotificationConfig and background_tasks:
            print("Running in NON-BLOCKING (webhook) mode.")
            # Add the long-running job to the background
            background_tasks.add_task(
                self._do_summarization_and_notify,
                message,
                config.pushNotificationConfig.url,
                config.pushNotificationConfig.token
            )
            # Immediately return a "working" response
            task_id = message.taskId or str(uuid4())
            context_id = str(uuid4())
            return TaskResult(
                id=task_id,
                contextId=context_id,
                status=TaskStatus(state="working")
            )

        # --- BLOCKING LOGIC (The code you already had) ---
        print("Running in BLOCKING mode.")
        task_id = message.taskId or str(uuid4())
        context_id = str(uuid4())
        extracted_url = self._find_youtube_url_in_message(message)
        if not extracted_url: raise ValueError("No valid YouTube URL found in the message.")
        video_id = self._get_video_id(extracted_url)
        if not video_id: raise ValueError("Could not extract a valid video ID from the URL.")
        try:
            print(f"[{task_id}] Fetching transcript for video ID: {video_id}...")
            transcript = self._get_transcript_from_api(video_id)
            print(f"[{task_id}] Summarizing transcript...")
            summary = await self._summarize_text(transcript)
            response_message = A2AMessage(role="agent", parts=[MessagePart(kind="text", text=summary)], taskId=task_id)
            artifacts = [Artifact(name="summary", parts=[MessagePart(kind="text", text=summary)]), Artifact(name="full_transcript", parts=[MessagePart(kind="text", text=transcript)])]
            return TaskResult(id=task_id, contextId=context_id, status=TaskStatus(state="completed", message=response_message), artifacts=artifacts, history=[message, response_message])
        except ValueError as e:
            error_text = str(e)
            error_message = A2AMessage(role="agent", parts=[MessagePart(kind="text", text=error_text)], taskId=task_id)
            return TaskResult(id=task_id, contextId=context_id, status=TaskStatus(state="failed", message=error_message), history=[message, error_message])

    # ... The rest of your agent file (_find_youtube_url_in_message, _get_video_id, etc.) remains the same ...
    def _find_youtube_url_in_message(self, message: A2AMessage) -> Optional[str]:
        url_pattern = re.compile(r'(https?://(www\.)?(youtube\.com/watch\?v=|youtu\.be/)[\w\-=&]+)')
        for part in reversed(message.parts):
            if part.kind == "text" and part.text:
                match = url_pattern.search(part.text)
                if match: return match.group(0)
            elif part.kind == "data" and isinstance(part.data, list):
                for item in reversed(part.data):
                    if isinstance(item, dict) and item.get("kind") == "text" and item.get("text"):
                        match = url_pattern.search(item["text"])
                        if match: return match.group(0)
        return None
    
    async def _summarize_text(self, transcript: str) -> str:
        prompt = f"Please summarize the following transcript:\n\n{transcript}"
        system_prompt = "You are an expert at summarizing YouTube video transcripts. Provide a concise, easy-to-read summary that captures the key points. Crucially, the entire response must be in plain text, with no Markdown formatting (no headers, bold text, or lists)."
        if self.provider == "google":
            full_prompt = f"{system_prompt}\n\n{prompt}"
            response = self.summarization_model.generate_content(full_prompt)
            return response.text
        elif self.provider == "openrouter":
            response = await self.http_client.post(url="https://openrouter.ai/api/v1/chat/completions", headers={"Authorization": f"Bearer {self.openrouter_api_key}"}, json={"model": self.openrouter_model, "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]})
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
            
    def _get_video_id(self, url: str) -> Optional[str]:
        if not url: return None
        query = urlparse(url)
        if query.hostname == 'youtu.be': return query.path[1:]
        if query.hostname in ('www.youtube.com', 'youtube.com'):
            if query.path == '/watch':
                p = parse_qs(query.query)
                return p.get('v', [None])[0]
            if query.path[:7] == '/embed/': return query.path.split('/')[2]
            if query.path[:3] == '/v/': return query.path.split('/')[2]
        return None

    def _get_transcript_from_api(self, video_id: str) -> str:
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            return " ".join([d['text'] for d in transcript_list])
        except (TranscriptsDisabled, NoTranscriptFound) as e:
            raise ValueError(f"A transcript is not available for this video. Captions may be disabled.")
        except Exception as e:
            raise ValueError("An unexpected error occurred while trying to get the video transcript.")