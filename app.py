# app.py (All-in-One, updated with deepface)

import streamlit as st
from PIL import Image
import os
from dotenv import load_dotenv
import google.generativeai as genai
import sqlite3
import numpy as np
import faiss
import io

# NEW: Import deepface
from deepface import DeepFace

# --- Part 1: Backend Logic (FaceManager Class) ---

class FaceManager:
    """ Manages all facial recognition using the lightweight deepface library. """
    def __init__(self, db_path='chatbot_data.db'):
        self.db_path = db_path
        self._initialize_database()
        
        self.known_face_encodings = []
        self.known_face_ids = []
        
        # We will use the 'Facenet' model, which produces 128-dimensional embeddings
        self.model_name = 'Facenet'
        self.index = faiss.IndexFlatL2(128)
        self.index = faiss.IndexIDMap(self.index)
        
        self._load_known_faces()

    def _get_db_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize_database(self):
        conn = self._get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS known_faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            embedding BLOB NOT NULL
        )''')
        conn.commit()
        conn.close()
        print("Database initialized.")

    def _load_known_faces(self):
        conn = self._get_db_connection()
        cursor = conn.execute('SELECT id, embedding FROM known_faces')
        encodings, ids = [], []
        for row in cursor:
            embedding = np.frombuffer(row['embedding'], dtype=np.float32)
            encodings.append(embedding)
            ids.append(row['id'])
        conn.close()
        
        if len(ids) > 0:
            self.known_face_encodings = np.array(encodings, dtype=np.float32)
            self.known_face_ids = np.array(ids)
            self.index.add_with_ids(self.known_face_encodings, self.known_face_ids)
            print(f"Loaded {len(self.known_face_ids)} known faces.")

    def add_face(self, image_bytes, name):
        """ UPDATED: Uses deepface to find a face and save its embedding. """
        try:
            # DeepFace works with image file paths, so we save bytes to a temp file
            with open("temp_face.jpg", "wb") as f:
                f.write(image_bytes)
            
            # Generate embedding using deepface
            embedding_objs = DeepFace.represent(img_path="temp_face.jpg", model_name=self.model_name, enforce_detection=True)
            new_encoding = np.array(embedding_objs[0]['embedding'], dtype=np.float32)

            os.remove("temp_face.jpg") # Clean up the temp file

            conn = self._get_db_connection()
            cursor = conn.execute(
                'INSERT INTO known_faces (name, embedding) VALUES (?, ?)',
                (name, new_encoding.tobytes())
            )
            new_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            self.index.add_with_ids(np.array([new_encoding]), np.array([new_id]))
            return True, f"Successfully added {name}."
        except ValueError:
            os.remove("temp_face.jpg")
            return False, "No face found or multiple faces detected. Please use a clear picture of one person."
        except Exception as e:
            if os.path.exists("temp_face.jpg"):
                os.remove("temp_face.jpg")
            return False, f"An error occurred: {e}"

    def recognize_faces(self, image_bytes):
        """ UPDATED: Uses deepface to recognize faces in a new image. """
        try:
            with open("temp_rec.jpg", "wb") as f:
                f.write(image_bytes)

            # Find all faces in the image and get their embeddings
            embedding_objs = DeepFace.represent(img_path="temp_rec.jpg", model_name=self.model_name, enforce_detection=False)
            os.remove("temp_rec.jpg")

            recognized_names = []
            conn = self._get_db_connection()
            
            for obj in embedding_objs:
                if not obj['face_confidence'] > 0.9: # Skip low-confidence detections
                    continue
                
                unknown_encoding = np.array([obj['embedding']], dtype=np.float32)
                distances, ids = self.index.search(unknown_encoding, k=1)
                
                # NOTE: The distance threshold for Facenet is different. ~1.1 is a good start.
                if ids[0][0] != -1 and distances[0][0] < 1.1:
                    matched_id = ids[0][0]
                    cursor = conn.execute('SELECT name FROM known_faces WHERE id = ?', (int(matched_id),))
                    row = cursor.fetchone()
                    if row and row['name'] not in recognized_names:
                        recognized_names.append(row['name'])
            
            conn.close()
            return recognized_names
        except Exception as e:
            if os.path.exists("temp_rec.jpg"):
                os.remove("temp_rec.jpg")
            print(f"Error during recognition: {e}")
            return []

# --- Part 2: Streamlit UI (No changes needed here, it remains the same) ---

# Load API Key
load_dotenv()
try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    st.error(f"🔴 Error initializing Google AI: {e}")
    st.stop()

@st.singleton
def get_face_manager():
    return FaceManager()

face_manager = get_face_manager()

st.set_page_config(page_title="AI Vision & Chat", layout="wide")
st.title("🧠 AI Vision Bot")
st.caption("Teach the bot who people are in the sidebar, then have a conversation about them below.")

with st.sidebar:
    st.header("Teach the Bot")
    st.info("Upload a clear picture of ONE person.")
    teach_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], key="teach_image")
    person_name = st.text_input("Person's Name", key="person_name")
    
    if st.button("Teach this person"):
        if teach_image and person_name:
            with st.spinner("Learning..."):
                image_bytes = teach_image.getvalue()
                success, message = face_manager.add_face(image_bytes, person_name)
                if success:
                    st.success(message)
                else:
                    st.error(f"Error: {message}")
        else:
            st.warning("Please upload an image and enter a name.")

st.header("Chat with the Vision Bot")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_session" not in st.session_state:
    st.session_state.chat_session = model.start_chat(history=[])

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Handle complex content (text + image)
        if isinstance(message["content"], list):
            for part in message["content"]:
                if isinstance(part, str):
                    st.markdown(part)
                elif isinstance(part, Image.Image):
                    st.image(part, width=200)
        else:
            st.markdown(message["content"])

uploaded_file = st.file_uploader("Upload a picture to discuss", type=["jpg", "jpeg", "png"])
if prompt := st.chat_input("Ask a question about the image or just chat..."):
    
    user_prompt_parts = []
    pil_image = None
    
    if uploaded_file is not None:
        pil_image = Image.open(uploaded_file)
        
        with st.spinner("Recognizing faces..."):
            image_bytes = uploaded_file.getvalue()
            names = face_manager.recognize_faces(image_bytes)
        
        recognition_context = f"(Context: I have recognized: {', '.join(names) if names else 'no one'}.) "
        final_prompt_text = recognition_context + prompt
        user_prompt_parts = [final_prompt_text, pil_image]

        with st.chat_message("user"):
            st.markdown(prompt)
            st.image(pil_image, width=200)
        
    else:
        user_prompt_parts = [prompt]
        with st.chat_message("user"):
            st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": [prompt, pil_image] if uploaded_file else prompt})
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_session.send_message(user_prompt_parts)
            ai_response = response.text
            st.markdown(ai_response)
    
    st.session_state.messages.append({"role": "assistant", "content": ai_response})
