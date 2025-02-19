from dotenv import load_dotenv
import os
from youtube_transcript_api import YouTubeTranscriptApi

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


def extract_transcript(youtube_url):
    """Extracts transcript text from a YouTube video."""
    try:
        video_id = youtube_url.split("=")[1]
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = ""
        for i in transcript:
            transcript_text += i['text']
        return transcript_text
    except Exception as e:
        print(f"Error: {e}")

