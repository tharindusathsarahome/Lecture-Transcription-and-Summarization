import os
import re
import time
import random
import winsound
import markdown
from datetime import datetime
from langchain_google_genai import GoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
import tenacity

class LectureSummarizer:
    def __init__(self, api_key: str, text_file: str):
        self.text_file = text_file
        self.llm = None
        self.docs = []
        self.splits = []
        self.map_prompt = None
        self.combine_prompt = None
        self.chain = None
        os.environ["GOOGLE_API_KEY"] = api_key

    def print_progress(self, message: str):
        """Print a progress message with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")

    def setup_llm(self):
        """Initialize the GoogleGenerativeAI model with rate limiting."""
        self.llm = self.RateLimitedGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
        self.print_progress("Initialized Gemini model with rate limiting")

    class RateLimitedGoogleGenerativeAI(GoogleGenerativeAI):
        """Custom class to handle rate limiting."""

        @tenacity.retry(
            wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
            stop=tenacity.stop_after_attempt(10),
            retry=tenacity.retry_if_exception_type(Exception),
            before_sleep=lambda retry_state: print(
                f"Rate limit hit, waiting {retry_state.next_action.sleep} seconds before retry {retry_state.attempt_number}"
            )
        )
        def generate_content(self, *args, **kwargs):
            time.sleep(random.uniform(2, 4))  # random delay
            return super().generate_content(*args, **kwargs)

    def read_transcript(self):
        """Read the lecture transcript file."""
        self.print_progress("Reading transcript file...")
        with open(f"{self.text_file}.txt", "r", encoding='utf-8') as file:
            transcript_text = file.read()
        self.print_progress(f"Successfully read file ({len(transcript_text)} characters)")
        return transcript_text

    def create_documents(self, text: str):
        """Convert text content into LangChain Document objects."""
        self.print_progress(f"Creating document object from text (length: {len(text)} characters)")
        self.docs = [Document(page_content=text)]

    def setup_text_splitter(self):
        """Configure the text splitter."""
        self.print_progress("Setting up text splitter with chunk size: 2000, overlap: 150")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=150,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", " ", ""]
        )
        self.splits = splitter.split_documents(self.docs)
        self.print_progress(f"Document split into {len(self.splits)} chunks")

    def setup_prompts(self):
        """Create map and combine prompts for the summarization chain."""
        self.print_progress("Setting up map and combine prompts")

        map_template = """Create a comprehensive and complete note from this lecture transcription segment:

        {text}

        KEY POINTS:"""

        combine_template = """Using the following segments of lecture notes, create a complete and well-structured study note:  

        {text}  

        The note should:  
        1. Start with a clear and concise introduction to the main topic.  
        2. Present all concepts and ideas in a logical order, ensuring clarity.  
        3. Highlight all important points and insights discussed in the lecture.  
        4. Clearly indicate topics or details specifically relevant for exams.  
        5. Conclude with the main takeaways and a summary of essential ideas.  

        Ensure the note is detailed yet simple to understand, retaining all critical information for effective exam preparation.  

        **DETAILED SUMMARY:** """

        self.map_prompt = PromptTemplate(template=map_template, input_variables=["text"])
        self.combine_prompt = PromptTemplate(template=combine_template, input_variables=["text"])

    def setup_chain(self):
        """Set up the summarization chain."""
        self.print_progress("Creating summarization chain")
        self.chain = load_summarize_chain(
            llm=self.llm,
            chain_type="map_reduce",
            map_prompt=self.map_prompt,
            combine_prompt=self.combine_prompt,
            verbose=False
        )

    def summarize(self):
        """Generate the summary."""
        try:
            self.print_progress("Starting summarization process")
            start_time = time.time()
            summary = self.chain.invoke(self.splits)
            end_time = time.time()
            self.print_progress(f"Summarization completed in {(end_time - start_time) / 60:.1f} minutes")
            return summary['output_text']
        except Exception as e:
            self.print_progress(f"Error during summarization: {e}")
            return f"Error generating summary: {str(e)}"

    def convert_to_html(self):
        """Convert the markdown summary to HTML."""
        self.print_progress("Converting summary to HTML...")
        try:
            input_file = f"{self.text_file}_summary.txt"

            with open(input_file, "r", encoding='utf-8') as file:
                summary_text = file.read()

            summary_text = re.sub(r'(\*\*.*\*\*)\n', r'\1\n\n', summary_text)

            html_content = markdown.markdown(summary_text)
            
            styled_html = f"""
            <html>
                <head>
                    <style>
                        body {{
                            font-family: Arial, sans-serif;
                            line-height: 1.6;
                            margin: 40px;
                            max-width: 800px;
                            margin: 40px auto;
                        }}
                        h1, h2, h3 {{
                            color: #2c3e50;
                            margin-top: 20px;
                        }}
                        p {{
                            margin-bottom: 15px;
                        }}
                        strong {{
                            color: #34495e;
                        }}
                    </style>
                </head>
                <body>
                    {html_content}
                </body>
            </html>
            """

            output_html = f"{input_file.replace('.txt', '.html')}"
            with open(output_html, "w", encoding='utf-8') as file:
                file.write(styled_html)
            self.print_progress(f"HTML summary saved to {output_html}")

            return output_html

        except Exception as e:
            self.print_progress(f"Error creating HTML: {str(e)}")

    def delete_summary_text(self):
        """Delete the summary text file."""
        try:
            os.remove(f"{self.text_file}_summary.txt")
            self.print_progress("Deleted summary text file")
        except FileNotFoundError:
            self.print_progress("Summary text file not found, skipping deletion")

    def run(self):
        """Main method to run the entire summarization process."""
        try:
            transcript_text = self.read_transcript()
            self.create_documents(transcript_text)
            self.setup_text_splitter()
            self.setup_prompts()
            self.setup_llm()
            self.setup_chain()
            summary = self.summarize()
            
            output_file = f"{self.text_file}_summary.txt"
            self.print_progress(f"Writing summary to {output_file}")
            with open(output_file, "w", encoding='utf-8') as file:
                file.write(summary)
            
            self.convert_to_html()
            
            self.print_progress("Process completed successfully!")
        except FileNotFoundError:
            self.print_progress(f"Error: Could not find the file {self.text_file}.txt")
        except Exception as e:
            self.print_progress(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    API_KEY = "AIzaSyD6N1TtmvIiiPLaYWWH8CHoxX6zLtOrzzI"
    TEXT_FILE = "D:/Desktop/UNI/~ACA - L3S1/CM3640 - Artificial Cognitive Systems/Recordings/2024-10-09 ACS Lec4_transcription"

    winsound.Beep(480, 500)
    summarizer = LectureSummarizer(api_key=API_KEY, text_file=TEXT_FILE)
    summarizer.run()
    summarizer.convert_to_html()
    summarizer.delete_summary_text()
    winsound.Beep(480, 300)
    winsound.Beep(600, 300)
    winsound.Beep(720, 300)
    winsound.Beep(600, 300)
    winsound.Beep(480, 300)
    winsound.Beep(360, 300)