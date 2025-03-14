import streamlit as st
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime

def send_message(sender_id, receiver_id, message_text, db_config):
    """Send a message from sender to receiver"""
    try:
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor()
        
        cur.execute("""
            INSERT INTO messages 
            (sender_id, receiver_id, message_text)
            VALUES (%s, %s, %s)
            RETURNING id
        """, (sender_id, receiver_id, message_text))
        
        message_id = cur.fetchone()[0]
        conn.commit()
        
        
        cur.execute("""
            INSERT INTO notifications (user_id, message, type)
            VALUES (%s, %s, 'new_message')
        """, (
            receiver_id,
            f"You have received a new message"
        ))
        
        conn.commit()
        return True, message_id
        
    except Exception as e:
        st.error(f"Error sending message: {e}")
        if conn:
            conn.rollback()
        return False, None
        
    finally:
        if conn:
            cur.close()
            conn.close()

def get_conversation_messages(user_id, other_user_id, db_config):
    """Get all messages between two users, sorted by creation time"""
    try:
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        cur.execute("""
            SELECT 
                m.id,
                m.sender_id,
                m.receiver_id,
                m.message_text,
                m.is_read,
                m.created_at,
                u_sender.full_name AS sender_name,
                u_receiver.full_name AS receiver_name
            FROM messages m
            JOIN users u_sender ON m.sender_id = u_sender.id
            JOIN users u_receiver ON m.receiver_id = u_receiver.id
            WHERE 
                (m.sender_id = %s AND m.receiver_id = %s)
                OR
                (m.sender_id = %s AND m.receiver_id = %s)
            ORDER BY m.created_at
        """, (user_id, other_user_id, other_user_id, user_id))
        
        messages = cur.fetchall()
        
        
        cur.execute("""
            UPDATE messages
            SET is_read = TRUE
            WHERE receiver_id = %s AND sender_id = %s AND is_read = FALSE
        """, (user_id, other_user_id))
        
        conn.commit()
        return messages
        
    except Exception as e:
        st.error(f"Error fetching messages: {e}")
        return []
        
    finally:
        if conn:
            cur.close()
            conn.close()

def get_user_conversations(user_id, db_config):
    """Get a list of conversations for a user"""
    try:
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        
        cur.execute("""
            WITH recent_messages AS (
                SELECT 
                    CASE 
                        WHEN sender_id = %s THEN receiver_id 
                        ELSE sender_id 
                    END AS other_user_id,
                    MAX(created_at) as last_message_time
                FROM messages
                WHERE sender_id = %s OR receiver_id = %s
                GROUP BY other_user_id
            )
            SELECT 
                rm.other_user_id,
                u.full_name as other_user_name,
                u.user_type as other_user_type,
                rm.last_message_time,
                (
                    SELECT COUNT(*) 
                    FROM messages 
                    WHERE 
                        receiver_id = %s AND 
                        sender_id = rm.other_user_id AND 
                        is_read = FALSE
                ) as unread_count
            FROM recent_messages rm
            JOIN users u ON rm.other_user_id = u.id
            ORDER BY rm.last_message_time DESC
        """, (user_id, user_id, user_id, user_id))
        
        conversations = cur.fetchall()
        return conversations
        
    except Exception as e:
        st.error(f"Error fetching conversations: {e}")
        return []
        
    finally:
        if conn:
            cur.close()
            conn.close()

def get_potential_recipients(user_id, user_type, db_config):
    """Get a list of potential message recipients based on user type"""
    try:
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        
        if user_type == 'student':
            cur.execute("""
                SELECT DISTINCT 
                    u.id,
                    u.full_name
                FROM supervisor_requests sr
                JOIN users u ON sr.supervisor_id = u.id
                WHERE sr.student_id = %s
                ORDER BY u.full_name
            """, (user_id,))
        
        elif user_type == 'supervisor':
            cur.execute("""
                SELECT DISTINCT 
                    u.id,
                    u.full_name
                FROM supervisor_requests sr
                JOIN users u ON sr.student_id = u.id
                WHERE sr.supervisor_id = %s
                ORDER BY u.full_name
            """, (user_id,))
        
        recipients = cur.fetchall()
        return recipients
        
    except Exception as e:
        st.error(f"Error fetching potential recipients: {e}")
        return []
        
    finally:
        if conn:
            cur.close()
            conn.close()

def display_messages_tab(db_config):
    """Display the messages tab UI for any user type"""
    if not st.session_state.get('authenticated', False):
        st.error("Please login to access this page")
        return
    
    user_id = st.session_state.user['id']
    user_type = st.session_state.user_type
    
    
    if 'active_conversation' not in st.session_state:
        st.session_state.active_conversation = None
    if 'new_message_recipient' not in st.session_state:
        st.session_state.new_message_recipient = None
    
    
    col1, col2 = st.columns([1, 3])
    
    
    with col1:
        st.subheader("Conversations")
        
        
        if st.button("➕ New Message", use_container_width=True):
            st.session_state.active_conversation = None
            st.session_state.new_message_recipient = True
            st.rerun()
        
        
        conversations = get_user_conversations(user_id, db_config)
        
        
        for conversation in conversations:
            other_user = conversation['other_user_name']
            unread = conversation['unread_count']
            
            
            button_label = other_user
            if unread > 0:
                button_label = f"{other_user} ({unread} new)"
            
           
            if st.button(button_label, key=f"conv_{conversation['other_user_id']}", use_container_width=True):
                st.session_state.active_conversation = conversation['other_user_id']
                st.session_state.new_message_recipient = None
                st.rerun()
    
    
    with col2:
        
        if st.session_state.new_message_recipient:
            st.subheader("New Message")
            
            
            recipients = get_potential_recipients(user_id, user_type, db_config)
            
            if not recipients:
                st.info("You don't have any contacts yet. For students, you can message supervisors after sending a request. For supervisors, you can message students who have sent you requests.")
            else:
                
                recipient_options = {r['full_name']: r['id'] for r in recipients}
                selected_name = st.selectbox("Select recipient:", list(recipient_options.keys()))
                selected_id = recipient_options[selected_name]
                
                
                message_text = st.text_area("Message:", height=100)
                
                
                if st.button("Send", use_container_width=True):
                    if message_text.strip():
                        success, _ = send_message(user_id, selected_id, message_text, db_config)
                        if success:
                            st.success("Message sent!")
                            st.session_state.active_conversation = selected_id
                            st.session_state.new_message_recipient = None
                            st.rerun()
                    else:
                        st.warning("Please enter a message")
        
        
        elif st.session_state.active_conversation:
            other_user_id = st.session_state.active_conversation
            
            
            conn = psycopg2.connect(**db_config)
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute("SELECT full_name FROM users WHERE id = %s", (other_user_id,))
            other_user = cur.fetchone()
            cur.close()
            conn.close()
            
            if other_user:
                st.subheader(f"Conversation with {other_user['full_name']}")
                
               
                messages = get_conversation_messages(user_id, other_user_id, db_config)
                
                
                st.markdown("""
                    <style>
                    .message-container {
                        height: 400px;
                        overflow-y: auto;
                        padding: 10px;
                        border: 1px solid #f0f0f0;
                        border-radius: 5px;
                        margin-bottom: 10px;
                    }
                    </style>
                """, unsafe_allow_html=True)
                
                
                st.markdown('<div class="message-container">', unsafe_allow_html=True)
                
                
                for msg in messages:
                    is_from_me = msg['sender_id'] == user_id
                    
                   
                    timestamp = msg['created_at'].strftime("%Y-%m-%d %H:%M")
                    
                    
                    if is_from_me:
                        st.markdown(f"""
                            <div style="margin-bottom: 10px; text-align: right;">
                                <div style="display: inline-block; background-color: #e3f2fd; padding: 10px; border-radius: 10px; max-width: 80%;">
                                    <div style="word-wrap: break-word;">{msg['message_text']}</div>
                                    <div style="font-size: 0.8em; color: #666; margin-top: 5px;">{timestamp}</div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                            <div style="margin-bottom: 10px; text-align: left;">
                                <div style="display: inline-block; background-color: #f1f1f1; padding: 10px; border-radius: 10px; max-width: 80%;">
                                    <div style="word-wrap: break-word;">{msg['message_text']}</div>
                                    <div style="font-size: 0.8em; color: #666; margin-top: 5px;">{timestamp}</div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                
                new_message = st.text_area("Message:", key="new_msg_input", height=100)
                
                
                if st.button("Send", key="send_in_conversation", use_container_width=True):
                    if new_message.strip():
                        success, _ = send_message(user_id, other_user_id, new_message, db_config)
                        if success:
                            st.success("Message sent!")
                            st.rerun()
                    else:
                        st.warning("Please enter a message")
            else:
                st.error("User not found")
        
        
        else:
            st.info("Select a conversation or start a new message")

if __name__ == "__main__":
    from auth_app import DB_CONFIG
    display_messages_tab(DB_CONFIG)