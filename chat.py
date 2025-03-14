import os
from openai import OpenAI
from dotenv import load_dotenv
import streamlit as st
import time
import uuid


load_dotenv()





api_key = st.secrets["OPENAI_API_KEY"] if hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

class SupervisorMatcherChatbot:
    def __init__(self):
        
        self.system_message = """
        You are a helpful assistant for the Research Supervisor Matcher system. 
        Your purpose is to help students navigate the platform and understand how to:
        1. Submit effective project proposals
        2. Understand how the matching algorithm works
        3. Interpret matching scores and results
        4. Contact and request supervision from professors
        5. Understand the status of their requests

        The Research Supervisor Matcher uses an advanced NLP algorithm that matches students with supervisors based on:
        - Research interests alignment (30% weight)
        - Methodology compatibility (25% weight)
        - Technical skills matching (20% weight)
        - Domain knowledge (15% weight)
        - Project type compatibility (10% weight)

        Students can submit up to 3 different project proposals, and each will be matched separately.
        Supervisors have capacity limits, and the system will indicate if a supervisor has reached capacity.
        
        Provide concise, accurate guidance about the system's features. If a student asks 
        about something unrelated to the system, politely redirect them to university support 
        services as appropriate.
        """
        
        
        if "chatbot_messages" not in st.session_state:
            st.session_state.chatbot_messages = [
                {"role": "system", "content": self.system_message},
                {"role": "assistant", "content": "Hello! I'm your Research Supervisor Matcher assistant. How can I help you with finding a research supervisor today?"}
            ]
        
        
        if "message_feedback" not in st.session_state:
            st.session_state.message_feedback = {}
    
    def get_messages_history(self):
        """Return the current conversation history"""
        return st.session_state.chatbot_messages
    
    def add_message(self, role, content):
        """Add a message to the conversation history"""
        message_id = str(uuid.uuid4())
        st.session_state.chatbot_messages.append({"role": role, "content": content, "id": message_id})
        return message_id
    
    def generate_response(self, user_input):
        """Generate a response using the OpenAI API"""
        
        # First we check if API key is properly loaded
        if not client.api_key:
            return "Error: OpenAI API key not found. Please contact the system administrator.", "error"
        
        
        user_msg_id = self.add_message("user", user_input)
        
        try:
            
            api_messages = []
            for msg in self.get_messages_history():
                if "role" in msg and "content" in msg:  
                    api_msg = {"role": msg["role"], "content": msg["content"]}
                    api_messages.append(api_msg)
            
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=api_messages,
                max_tokens=500,
                temperature=0.7,
            )
            
            
            assistant_message = response.choices[0].message.content
            
            
            assistant_msg_id = self.add_message("assistant", assistant_message)
            
            return assistant_message, assistant_msg_id
            
        except Exception as e:
            error_message = f"Sorry, I encountered an error: {str(e)}"
            assistant_msg_id = self.add_message("assistant", error_message)
            return error_message, assistant_msg_id
    
    def clear_conversation(self):
        """Reset the conversation history"""
        st.session_state.chatbot_messages = [
            {"role": "system", "content": self.system_message},
            {"role": "assistant", "content": "Hello! I'm your Research Supervisor Matcher assistant. How can I help you with finding a research supervisor today?", "id": str(uuid.uuid4())}
        ]
        st.session_state.message_feedback = {}
    
    def record_feedback(self, message_id, feedback):
        """Record user feedback for a message"""
        st.session_state.message_feedback[message_id] = feedback
        
    def display_chat_interface(self):
        """Display the chat interface in Streamlit"""
        st.subheader("Chatbot Guidance")
        
        
        st.markdown("""
        <style>
        .user-message {
            background-color: #E9ECEF;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 0.5rem;
            text-align: right;
            position: relative;
            margin-left: 20%;
            word-wrap: break-word;
            overflow-wrap: break-word;
        }
        .assistant-message {
            background-color: #4F46E5;
            color: white;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 0.5rem;
            position: relative;
            margin-right: 20%;
            word-wrap: break-word;
            overflow-wrap: break-word;
        }
        .chat-input-container {
            display: flex;
            align-items: center;
            margin-top: 1rem;
        }
        .chat-input {
            flex-grow: 1;
            border-radius: 0.5rem;
            border: 1px solid #E2E8F0;
            padding: 0.75rem;
            padding-left: 40px; /* Make space for the emoji */
        }
        .send-button {
            margin-left: 0.5rem;
        }
        .chat-container {
            display: flex;
            flex-direction: column;
            padding: 1rem;
            background-color: white;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            border: 1px solid #E2E8F0;
            min-height: 350px;
            max-height: 500px;
            overflow-y: auto;
        }
        .suggestion-item {
            margin: 10px 0;
            padding: 8px;
            background-color: #f8f9fa;
            border-radius: 4px;
            cursor: pointer;
            transition: background-color 0.2s;
        }
        .suggestion-item:hover {
            background-color: #e9ecef;
        }
        .feedback-button {
            border: none;
            background: transparent;
            font-size: 1rem;
            cursor: pointer;
            opacity: 0.7;
            transition: opacity 0.2s;
        }
        .feedback-button:hover {
            opacity: 1;
        }
        .feedback-active {
            opacity: 1;
            color: #4F46E5;
        }
        .user-emoji {
            position: absolute;
            right: -25px;
            top: 50%;
            transform: translateY(-50%);
            font-size: 1.2rem;
        }
        .assistant-emoji {
            position: absolute;
            left: -25px;
            top: 50%;
            transform: translateY(-50%);
            font-size: 1.2rem;
        }
        .input-emoji {
            position: absolute;
            left: 10px;
            top: 50%;
            transform: translateY(-50%);
            font-size: 1.2rem;
            z-index: 10;
        }
        .typing-animation {
            overflow: hidden;
            white-space: normal; /* Changed from nowrap to normal */
            animation: fade-in 0.5s;
        }
        @keyframes fade-in {
            0% { opacity: 0; }
            100% { opacity: 1; }
        }
        /* Fix the input and send button alignment */
        .stHorizontal {
            flex-wrap: nowrap !important;
            gap: 0.5rem !important;
        }
        /* Fix the empty container issue */
        .st-emotion-cache-ue6h4q {
            display: none;
        }
        /* Removing padding around main chat container */
        div.element-container div.stMarkdown {
            padding-bottom: 0px !important;
        }
        /* Fix for long messages */
        .stMarkdown p {
            word-wrap: break-word !important;
            white-space: normal !important;
        }
        /* Button style to match logout button */
        .stButton > button {
            background-color: #5146E5 !important;
            color: white !important;
            border: none !important;
            border-radius: 25px !important;
            padding: 0.5rem 1rem !important;
            font-weight: 500 !important;
            transition: background-color 0.3s !important;
        }
        .stButton > button:hover {
            background-color: #4338CA !important;
            border: none !important;
        }
        .stDownloadButton > button {
            background-color: #5146E5 !important;
            color: white !important;
            border: none !important;
            border-radius: 25px !important;
            padding: 0.5rem 1rem !important;
            font-weight: 500 !important;
            transition: background-color 0.3s !important;
        }
        .stDownloadButton > button:hover {
            background-color: #4338CA !important;
            border: none !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        
        chat_container = st.container()
        
        with chat_container:
            
            for i, message in enumerate(st.session_state.chatbot_messages):
                if message["role"] == "system":
                    continue  
                
                msg_id = message.get("id", str(uuid.uuid4()))
                
                if message["role"] == "assistant":
                    st.markdown(f'<div class="assistant-message"><span class="assistant-emoji">🤖</span><span class="typing-animation">{message["content"]}</span></div>', unsafe_allow_html=True)
                    
                    
                    if i > 1:
                        feedback = st.session_state.message_feedback.get(msg_id, None)
                        helpful_class = "feedback-active" if feedback == "helpful" else ""
                        not_helpful_class = "feedback-active" if feedback == "not_helpful" else ""
                        
                        col1, col2, col3 = st.columns([1, 1, 10])
                        with col1:
                            if st.button("👍", key=f"helpful_{msg_id}", help="This response was helpful"):
                                self.record_feedback(msg_id, "helpful")
                                st.rerun()
                        with col2:
                            if st.button("👎", key=f"not_helpful_{msg_id}", help="This response was not helpful"):
                                self.record_feedback(msg_id, "not_helpful")
                                st.rerun()
                
                elif message["role"] == "user":
                    st.markdown(f'<div class="user-message">{message["content"]}<span class="user-emoji">👨‍🎓</span></div>', unsafe_allow_html=True)
            
            
            if len(st.session_state.chatbot_messages) <= 2:
                st.markdown("""
                    <div style="margin-top: 20px; color: #666; text-align: center;">
                        <p>Some questions you might ask:</p>
                        <ul style="list-style-type: none; padding: 0;">
                            <li class="suggestion-item">How do I create an effective project proposal?</li>
                            <li class="suggestion-item">What factors affect the matching algorithm?</li>
                            <li class="suggestion-item">How do I interpret the matching scores?</li>
                            <li class="suggestion-item">Can I request multiple supervisors?</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)
        
        
        user_input_container = st.container()
        with user_input_container:
            cols = st.columns([8, 2])
            with cols[0]:
                st.markdown('<div style="position: relative; width: 100%;">', unsafe_allow_html=True)
                st.markdown('<span class="input-emoji">👨‍🎓</span>', unsafe_allow_html=True)
                user_input = st.text_input("Type your question here...", 
                                        key="chat_input", 
                                        label_visibility="collapsed")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with cols[1]:
                submit_button = st.button("Send", key="submit_chat", use_container_width=True)
        
        
        col1, col2 = st.columns(2)
        with col1:
            clear_button = st.button("🗑️ Clear Conversation", key="clear_chat", use_container_width=True)
        with col2:
            
            if len(st.session_state.chatbot_messages) > 2:  
                conversation_text = "\n\n".join(
                    [f"You: {msg['content']}" if msg['role'] == 'user' else f"Assistant: {msg['content']}" 
                    for msg in st.session_state.chatbot_messages if msg['role'] != 'system']
                )
                st.download_button(
                    label="📥 Export Conversation",
                    data=conversation_text,
                    file_name=f"supervision_chat_{time.strftime('%Y%m%d-%H%M%S')}.txt",
                    mime="text/plain",
                    key="export_chat",
                    use_container_width=True
                )
        
       
        if submit_button and user_input:
            response, msg_id = self.generate_response(user_input)
            

        
        if clear_button:
            self.clear_conversation()
            st.rerun()
            
        # FAQ Section
        with st.expander("Frequently Asked Questions"):
            st.markdown("""
            ### Common Questions
            
            **Q: How do I create an effective project proposal?**  
            A: Focus on clear research objectives, methodology, and required technical skills. Be specific about your research interests and technical requirements. Include the project type (Research-based, Theoretical, Industry-focused, etc.) to improve matching accuracy.
            
            **Q: What makes a good match with a supervisor?**  
            A: Research interest alignment (30%), methodology compatibility (25%), technical skills (20%), domain knowledge (15%), and project type compatibility (10%) are the key factors in our matching algorithm.
            
            **Q: Can I apply to multiple supervisors?**  
            A: Yes, you can request supervision from multiple professors. Our system allows you to see all potential matches and choose the best fits for your research interests.
            
            **Q: What happens after I request supervision?**  
            A: Supervisors will review your request and either accept or decline it. You can check the status in the "My Requests" tab. Each supervisor has a capacity limit, so some may decline if they have reached their maximum student count.
            
            **Q: How do I interpret the matching scores?**  
            A: Scores range from 0 to 1, with higher scores indicating better alignment. Scores above 0.7 suggest excellent compatibility, 0.5-0.7 is good, and below 0.5 may indicate limited alignment with your research interests.
            """)

def display_chatbot_guidance():
    """Function to initialize and display the chatbot in the app"""
    chatbot = SupervisorMatcherChatbot()
    chatbot.display_chat_interface()

if __name__ == "__main__":
    
    st.title("Research Supervisor Matcher")
    display_chatbot_guidance()