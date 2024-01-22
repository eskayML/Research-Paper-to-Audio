import streamlit as st
import pyttsx3
import tempfile
import PyPDF2
from huggingface_hub import InferenceClient

page_bg_img = """
<style>
.stApp {
background: linear-gradient(  #eee 38%, #ccc 68%);
}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

st.title("Summarize & Listen to your Academic Materials on the Fly.")

uploaded_pdf = st.file_uploader("Upload a research Paper", type="pdf")
full_text = None
MODEL_NAME = "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO"
client = InferenceClient(MODEL_NAME)


DETAILED_SUMMARIZATION_PROMPT = """
<INST>You are a very powerful summarization engine for summarizing academic contents,
now you are to  summarize the following text you are going to be provided which is from a document, make sure to understand 
all improperly parsed text and actually parse them properly , also make sure that your final summarization is very coherent and understandable by a student and is under 4000 words ,
 also the length of the summarized text should be less than the original provided text,
 if you are provided with a text that includes unnecessary items that do not contribute value to the book like preface about the author, do not include them in the summarization

  Your summary should be concise and should accurately and objectively communicate the key points of the paper.
   You should not include any personal opinions or interpretations in your summary but rather focus on 
   objectively presenting the information from the paper. Your summary should be written in your own words 
   and should not include any direct quotes from the paper. Please ensure that your summary is clear, 
   concise, and accurately reflects the content of the original paper.
 do not go out of context of the words provided. 
 Now here is your provided text :
</INST>
"""


with st.spinner("Extracting Text..."):
    if uploaded_pdf is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_pdf.read())
        with open(tfile.name, "rb") as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            num_pages = len(pdf_reader.pages)

            # Get text from all pages
            full_text = ""
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                full_text += page_text

            # truncating the full text at 25k characters
            full_text = full_text if len(full_text) < 100000 else full_text[:100000]
            # print(full_text)
            st.success("Text Extracted Successfully!!!")


###################################################################################


def synthesize_text_to_audio(text):
    engine = pyttsx3.init()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
        temp_file_path = temp_file.name
        engine.save_to_file(text, temp_file_path)  # Save the audio to a temporary file

    engine.runAndWait()
    sound_file = open(temp_file_path, "rb")  # Open the saved audio file for reading
    return sound_file



summarized_text = None
if full_text:
    with st.spinner("Summarizing Text Content..."):
        summarized_text = client.text_generation(
            DETAILED_SUMMARIZATION_PROMPT + full_text,
            max_new_tokens=4096,
            temperature=0.2,
            top_p=0.8,
        )
        print(summarized_text)

if summarized_text:
    with st.spinner('Synthesizing to Audio...'):
        st.audio(synthesize_text_to_audio(summarized_text))
        
