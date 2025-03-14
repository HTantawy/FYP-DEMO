import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from student_supervisor import AdvancedSupervisorMatcher, visualize_results, generate_report
from supervisor_view import view_supervisor_profile
from chat import display_chatbot_guidance  
import json
from datetime import datetime
import plotly.graph_objects as go
import psycopg2
from psycopg2.extras import RealDictCursor
import os
from dotenv import load_dotenv
from messaging import display_messages_tab

load_dotenv()


DB_CONFIG = {
    'dbname': os.getenv('POSTGRES_DATABASE'),
    'user': os.getenv('POSTGRES_USER'),
    'password': os.getenv('POSTGRES_PASSWORD'),
    'host': os.getenv('POSTGRES_HOST'),
    'port': os.getenv('POSTGRES_PORT', '5432'),
    'sslmode': 'require'  
}

# Initialize the matcher
@st.cache_resource
def load_matcher():
    return AdvancedSupervisorMatcher()

def get_supervisors_from_db():
    """Fetch supervisors from database (including their maximum capacity)"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        cur.execute("""
            SELECT 
                u.id,
                u.full_name as name,
                sp.research_interests as interests,
                sp.department,
                sp.expertise,    
                sp.preferred_projects as project_types,
                sp.max_capacity as max_capacity
            FROM users u
            JOIN supervisor_profiles sp ON u.id = sp.user_id
            WHERE u.user_type = 'supervisor'
        """)
        
        supervisors = cur.fetchall()
        return supervisors
        
    except Exception as e:
        st.error(f"Database error: {e}")
        return []
        
    finally:
        if conn:
            cur.close()
            conn.close()

def save_supervisor_request(student_id, supervisor_id, project_data, match_score):
    """Save a new supervisor request"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        cur.execute("""
            INSERT INTO supervisor_requests 
            (student_id, supervisor_id, project_title, project_description, matching_score)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (student_id, supervisor_id) 
            DO UPDATE SET 
                project_title = EXCLUDED.project_title,
                project_description = EXCLUDED.project_description,
                matching_score = EXCLUDED.matching_score,
                status = 'pending',
                updated_at = NOW()
            RETURNING id
        """, (
            student_id,
            supervisor_id,
            project_data['title'],
            project_data['description'],
            match_score
        ))
        
        request_id = cur.fetchone()[0]
        conn.commit()
        return True
        
    except Exception as e:
        st.error(f"Error saving request: {e}")
        return False
        
    finally:
        if conn:
            cur.close()
            conn.close()

def create_match_visualization(matches):
    """Create visualization for matching results"""
    categories = ['Research Alignment', 'Methodology Match', 'Technical Skills', 'Domain Knowledge', 'Project Type Match']
    
    fig = go.Figure()
    
    for i, match in enumerate(matches[:3]):  # Top 3 matches
        scores = [
            match['detailed_scores']['research_alignment'],
            match['detailed_scores']['methodology_match'],
            match['detailed_scores']['technical_skills'],
            match['detailed_scores']['domain_knowledge'],
            match['detailed_scores'].get('project_type_match', 0.0) 
        ]
        
        fig.add_trace(go.Bar(
            name=match['supervisor_name'],
            x=categories,
            y=scores,
            text=[f'{score:.2f}' for score in scores],
            textposition='auto',
        ))

    fig.update_layout(
        title='Top 3 Matches - Score Breakdown',
        yaxis_title='Score',
        barmode='group',
        showlegend=True,
        height=500
    )
    
    return fig

def get_student_requests(student_id):
    """Get all requests made by a student"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        cur.execute("""
            SELECT 
                sr.*,
                u.full_name as supervisor_name,
                sp.department,
                sp.research_interests
            FROM supervisor_requests sr
            JOIN users u ON sr.supervisor_id = u.id
            JOIN supervisor_profiles sp ON u.id = sp.user_id
            WHERE sr.student_id = %s
            ORDER BY sr.created_at DESC
        """, (student_id,))
        
        requests = cur.fetchall()
        return requests
        
    except Exception as e:
        st.error(f"Error fetching requests: {e}")
        return []
        
    finally:
        if conn:
            cur.close()
            conn.close()

def save_match_history(student_id, supervisor_id, match_data):
    """Save match details to the matching_history table"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        cur.execute("""
            INSERT INTO matching_history 
            (student_id, supervisor_id, final_score, research_alignment, methodology_match, technical_skills, domain_knowledge, project_type_match, matching_skills)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            student_id,
            supervisor_id,
            match_data['final_score'],
            match_data['detailed_scores']['research_alignment'],
            match_data['detailed_scores']['methodology_match'],
            match_data['detailed_scores']['technical_skills'],
            match_data['detailed_scores']['domain_knowledge'],
            match_data['detailed_scores'].get('project_type_match', 0.0),
            json.dumps(match_data.get('matching_skills', []))
        ))

        conn.commit()
    except Exception as e:
        st.error(f"Error saving match history: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            cur.close()
            conn.close()


def check_supervisor_capacity(supervisor_id, max_capacity):
    """
    Check if the supervisor has reached their capacity.
    Returns True if the number of active requests (pending or accepted) is less than max_capacity.
    """
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("""
            SELECT COUNT(*) 
            FROM supervisor_requests
            WHERE supervisor_id = %s 
              AND status IN ('pending', 'accepted')
        """, (supervisor_id,))
        count = cur.fetchone()[0]
        return count < max_capacity
    except Exception as e:
        st.error(f"Error checking supervisor capacity: {e}")
        return False
    finally:
        if conn:
            cur.close()
            conn.close()

def init_session_state():
    if 'project_data' not in st.session_state:
        st.session_state.project_data = [
            {
                'title': '',
                'description': '',
                'technical_requirements': [],
                'methodology': 'Quantitative',
                'project_type': []
            },
            {
                'title': '',
                'description': '',
                'technical_requirements': [],
                'methodology': 'Quantitative',
                'project_type': []
            },
            {
                'title': '',
                'description': '',
                'technical_requirements': [],
                'methodology': 'Quantitative',
                'project_type': []
            }
        ]
    if 'active_project' not in st.session_state:
        st.session_state.active_project = 0
    if 'matching_results' not in st.session_state:
        st.session_state.matching_results = None

def stylish_tab_navigation():
    """Create a stylish tab navigation system"""
    
    
    tabs = [
        {"name": "Search Supervisors", "icon": "🔍"},
        {"name": "My Requests", "icon": "📋"},
        {"name": "Messages", "icon": "✉️"},
        {"name": "Chatbot Guidance", "icon": "💬"}
    ]
    
    
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = "Search Supervisors"
    
    
    st.markdown("""
    <style>
        /* Custom tab styling */
        div.stButton > button {
            background-color: white;
            border: none;
            color: #718096;
            padding: 0.75rem 0;
            font-weight: 500;
            border-radius: 0;
            width: 100%;
            text-align: center;
        }
        
        div.stButton > button:hover {
            background-color: #f8fafc;
            color: #4051b5;
        }
        
        div.stButton > button:focus:not(:active) {
            border-color: transparent;
            box-shadow: none;
        }
        
        div.stButton > button[data-active="true"] {
            border-bottom: 3px solid #4051b5;
            color: #4051b5;
            font-weight: 600;
        }
        
        /* Tab container styling */
        [data-testid="stHorizontalBlock"] {
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            padding: 0;
            margin-bottom: 1.5rem;
        }
    </style>
    """, unsafe_allow_html=True)
    
    
    cols = st.columns(len(tabs))
    
   
    for i, tab in enumerate(tabs):
        active_status = "true" if st.session_state.active_tab == tab["name"] else "false"
        
        if cols[i].button(
            f"{tab['icon']} {tab['name']}", 
            key=f"tab_{i}",
            use_container_width=True,
            args=(active_status,)
        ):
            st.session_state.active_tab = tab["name"]
            st.rerun()
    
    return st.session_state.active_tab

def show_search_page():
    """Show the supervisor search and matching page"""
    init_session_state()
    
    
    if 'viewing_profile' in st.session_state:
        view_supervisor_profile(
            st.session_state.viewing_profile,
            st.session_state.viewing_name,
            DB_CONFIG
        )
        if st.button("← Back to Search Results"):
            del st.session_state.viewing_profile
            del st.session_state.viewing_name
            st.rerun()
        return

    matcher = load_matcher()
    supervisors = get_supervisors_from_db()
    
    
    st.markdown("""
        <style>
        .project-card {
            background-color: white;
            padding: 2rem;
            border-radius: 1rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            margin-bottom: 2rem;
        }
        
        .stTextInput input,
        .stTextArea textarea,
        .stSelectbox select {
            border: 1px solid #E2E8F0;
            border-radius: 0.5rem;
            padding: 0.75rem;
            background-color: #F7FAFC;
        }
        
        .stButton > button {
            background-color: #4F46E5 !important;
            color: white !important;
            border-radius: 0.5rem !important;
            padding: 0.5rem 1rem !important;
            font-weight: 500 !important;
            border: none !important;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1) !important;
            transition: all 200ms ease-in-out !important;
            font-size: 0.875rem !important;
            width: auto !important;
            min-width: 120px !important;
            height: 38px !important;
            line-height: 1.1 !important;
        }
        
        .stButton > button:hover {
            background-color: #4338CA !important;
            transform: translateY(-1px);
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("Project Proposals")
    
    
    tab1, tab2, tab3 = st.tabs(["Project 1", "Project 2", "Project 3"])

    def create_project_form(index):
        form_key = f"project_form_{index}"
        
        with st.form(key=form_key):
            student_name = st.text_input(
                "Your Name", 
                value=st.session_state.user.get('full_name', ''),
                disabled=True,
                key=f"name_{index}"
            )
            
            project_title = st.text_input(
                "Project Title",
                value=st.session_state.project_data[index].get('title', ''),
                key=f"title_{index}"
            )
            
            project_description = st.text_area(
                "Project Description",
                value=st.session_state.project_data[index].get('description', ''),
                height=200,
                placeholder="Describe your research project, including methodologies, technical requirements, and expected outcomes...",
                key=f"desc_{index}"
            )

            col1, col2 = st.columns(2)
            with col1:
                project_type_options = [
                    "Research-based", 
                    "Theoretical", "Industry-focused",
                    "Software Development", "Hardware/IoT",
                ]
                
                selected_project_type = st.multiselect(
                    "Project Type",
                    options=sorted(project_type_options),
                    default=st.session_state.project_data[index].get('project_type', []),
                    key=f"type_{index}"
                )
            
            with col2:
                tech_options = [
                    'Python', 'R', 'Machine Learning', 'Deep Learning', 'Statistical Analysis',
                    'Data Mining', 'NLP', 'Computer Vision', 'Blockchain', 'Cloud Computing',
                    'TensorFlow', 'PyTorch', 'Scikit-learn', 'Robotics', 'JavaScript',
                    'Transformers', 'IoT', 'Web Development', 'Mobile App Development',
                    'Database Systems', 'Node.js', 'Kotlin', 'Data Analysis', 'Reinforcement Learning', 'Game Development','Java','Neural Networks', 'Speech Recognition'
                ]
                
                selected_tech = st.multiselect(
                    'Technical Requirements',
                    options=sorted(tech_options),
                    default=st.session_state.project_data[index].get('technical_requirements', []),
                    key=f"tech_{index}"
                )
            
            methodology = st.selectbox(
                'Primary Research Methodology',
                ['Quantitative', 'Qualitative', 'Mixed Methods', 'Experimental'],
                index=['Quantitative', 'Qualitative', 'Mixed Methods', 'Experimental'].index(
                    st.session_state.project_data[index].get('methodology', 'Quantitative')
                ),
                key=f"method_{index}"
            )
            
            submitted = st.form_submit_button("Find Matching Supervisors")
            if submitted:
                if not project_description:
                    st.error("Please provide a project description")
                else:
                    st.session_state.project_data[index] = {
                        'title': project_title,
                        'description': project_description,
                        'technical_requirements': selected_tech,
                        'methodology': methodology,
                        'project_type': selected_project_type
                    }
                    
                    student_data = {
                        'student_name': student_name,
                        'project_title': project_title,
                        'project_description': (
                            f"{project_description}\n"
                            f"Technical requirements: {', '.join(selected_tech)}.\n"
                            f"Research methodology: {methodology}."
                            f"Project type: {', '.join(selected_project_type)}."
                        ),
                        'project_type': selected_project_type
                    }
                    
                    matches = matcher.match_supervisors(student_data, supervisors)
                    st.session_state.matching_results = matches
                    st.session_state.active_project = index
                    st.rerun()

    with tab1:
        create_project_form(0)
    
    with tab2:
        create_project_form(1)

    with tab3:
        create_project_form(2)    

    if st.session_state.matching_results:
        st.markdown(f"## Matching Results for Project {st.session_state.active_project + 1}")
        
        fig = create_match_visualization(st.session_state.matching_results)
        st.plotly_chart(fig, use_container_width=True)
        
        st.write("### Top Matches")
        for i, match in enumerate(st.session_state.matching_results[:3], 1):
            st.write(f"### #{i} Match Score: {match['final_score']:.3f}")
            supervisor = next(
                (s for s in supervisors if s['name'] == match['supervisor_name']), 
                None
            )
            if supervisor:
                with st.expander(f"View Details: {supervisor['name']}"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.write("**Department:**", supervisor['department'])
                        st.write("**Research Interests:**")
                        st.write(supervisor['interests'])
                        st.write("**Expertise Areas:**")
                        st.write(", ".join(supervisor['expertise']) if supervisor['expertise'] else "Not specified")
                        st.write("**Preferred Project Types:**")
                        st.write(", ".join(supervisor['project_types']) if supervisor.get('project_types') else "Not specified")
                    
                    with col2:
                        st.write("**Match Scores:**")
                        scores_df = pd.DataFrame({
                            'Metric': ['Research Alignment', 'Methodology Match', 
                                     'Technical Skills', 'Domain Knowledge', 'Project Type Match'],
                            'Score': [
                                match['detailed_scores']['research_alignment'],
                                match['detailed_scores']['methodology_match'],
                                match['detailed_scores']['technical_skills'],
                                match['detailed_scores']['domain_knowledge'],
                                match['detailed_scores'].get('project_type_match', 0.0)
                            ]
                        })
                        st.dataframe(scores_df)
                        
                        if match.get('matching_skills'):
                            st.write("**Matching Skills:**")
                            st.write(", ".join(match['matching_skills']))

                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("View Full Profile", key=f"profile_{supervisor['id']}"):
                                st.session_state.viewing_profile = supervisor['id']
                                st.session_state.viewing_name = supervisor['name']
                                st.rerun()
                        
                        with col2:
                            if st.button("Request Supervision", key=f"request_{supervisor['id']}"):
                                # Check supervisor capacity before saving the request
                                if not check_supervisor_capacity(supervisor['id'], supervisor['max_capacity']):
                                    st.error("Supervisor capacity is full. Please choose another supervisor.")
                                else:
                                    active_project = st.session_state.project_data[st.session_state.active_project]
                                    if save_supervisor_request(
                                        st.session_state.user['id'],
                                        supervisor['id'],
                                        active_project,
                                        match['final_score']
                                    ):
                                        save_match_history(
                                            st.session_state.user['id'],
                                            supervisor['id'],
                                            match
                                        )
                                        st.success(f"Request sent to {supervisor['name']}!")
                                    else:
                                        st.error("Failed to send request. Please try again.")

def show_requests_page():
    """Show the student's supervision requests"""
    st.subheader("My Supervision Requests")
    
    requests = get_student_requests(st.session_state.user['id'])
    
    if not requests:
        st.info("You haven't made any supervision requests yet.")
        return
    
    for request in requests:
        with st.expander(f"{request['project_title']} - {request['supervisor_name']}"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write("**Project Description:**")
                st.write(request['project_description'])
                st.write("**Supervisor Department:**", request['department'])
                st.write("**Research Interests:**")
                st.write(request['research_interests'])
                if request.get('project_types'):
                    st.write(", ".join(request['project_types']))
                else:
                    st.write("Not specified")
            
            with col2:
                status_colors = {
                    'pending': 'blue',
                    'accepted': 'green',
                    'rejected': 'red'
                }
                st.write("**Status:**", f":{status_colors[request['status']]}[{request['status'].upper()}]")
                st.write("**Matching Score:**", f"{request['matching_score']:.2f}")
                st.write("**Submitted:**", request['created_at'].strftime("%Y-%m-%d %H:%M"))
                if request['updated_at'] != request['created_at']:
                    st.write("**Last Updated:**", request['updated_at'].strftime("%Y-%m-%d %H:%M"))

def student_main():
    if not st.session_state.get('authenticated', False):
        st.error("Please login to access this page")
        return
    
    init_session_state()
    
    st.markdown("""
        <div style="background-color: #fff; 
                    padding: 1.5rem; 
                    border-radius: 12px; 
                    box-shadow: 0 4px 10px rgba(0,0,0,0.08); 
                    margin-bottom: 1.5rem;">
            <h1 style="font-size: 2.2rem; 
                    font-weight: 700; 
                    color: #1E3D59; 
                    margin: 0;">
                Student Dashboard
            </h1>
        </div>
    """, unsafe_allow_html=True)

    
    st.session_state.active_tab = stylish_tab_navigation()
    
   
    if st.session_state.active_tab == "Search Supervisors":
        show_search_page()
    elif st.session_state.active_tab == "My Requests":
        show_requests_page()
    elif st.session_state.active_tab == "Messages":  
        display_messages_tab(DB_CONFIG)    
    else:  
        display_chatbot_guidance()
    
   
    with st.sidebar:
        st.markdown("""
        <div style="
            background-color: white;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            margin-bottom: 1.5rem;
        ">
            <h3 style="
                color: #1E3D59;
                font-size: 1.2rem;
                margin-bottom: 0.8rem;
                font-weight: 600;
            ">About</h3>
            <p style="
                color: #4A5568;
                font-size: 0.9rem;
                line-height: 1.5;
            ">
                This tool uses advanced Natural Language Processing and Machine Learning 
                techniques to match students with potential research supervisors based on:
            </p>
            <ul style="
                color: #4A5568;
                font-size: 0.9rem;
                line-height: 1.5;
                padding-left: 1.5rem;
                margin-top: 0.5rem;
            ">
                <li>Research interests alignment</li>
                <li>Methodology compatibility</li>
                <li>Technical skill requirements</li>
                <li>Domain knowledge</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="
            background-color: white;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        ">
            <h3 style="
                color: #1E3D59;
                font-size: 1.2rem;
                margin-bottom: 0.8rem;
                font-weight: 600;
            ">How it works</h3>
            <ol style="
                color: #4A5568;
                font-size: 0.9rem;
                line-height: 1.5;
                padding-left: 1.5rem;
                margin-top: 0.5rem;
            ">
                <li>Enter your project details</li>
                <li>Specify technical requirements</li>
                <li>Choose research methodology</li>
                <li>Get matched with potential supervisors</li>
                <li>Review detailed matching scores</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
       
        if st.button("Logout", key="sidebar_logout_button", use_container_width=True):
            st.session_state.clear()
            st.rerun()

if __name__ == "__main__":
    student_main()