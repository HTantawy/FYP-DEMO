import streamlit as st
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime

def get_supervisor_profile(supervisor_id, db_config):
    """Fetch supervisor's complete profile"""
    try:
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        # Get basic profile
        cur.execute("""
            SELECT 
                u.full_name,
                u.email,
                sp.department,
                sp.research_interests,
                sp.expertise,
                sp.preferred_projects,
                sp.office_hours,
                sp.contact_preferences,
                sp.website_url,
                sp.bio
            FROM users u
            JOIN supervisor_profiles sp ON u.id = sp.user_id
            WHERE u.id = %s
        """, (supervisor_id,))
        
        profile = cur.fetchone()
        
        # Get publications
        cur.execute("""
            SELECT * FROM supervisor_publications
            WHERE supervisor_id = %s
            ORDER BY year DESC, title
        """, (supervisor_id,))
        publications = cur.fetchall()
        
        # Get supervised projects
        cur.execute("""
            SELECT * FROM supervised_projects
            WHERE supervisor_id = %s
            ORDER BY year DESC, title
        """, (supervisor_id,))
        projects = cur.fetchall()
        
        return {
            'profile': profile,
            'publications': publications,
            'projects': projects
        }
        
    except Exception as e:
        st.error(f"Error fetching profile: {e}")
        return None
    finally:
        if conn:
            cur.close()
            conn.close()

def update_supervisor_profile(supervisor_id, profile_data, db_config):
    """Update supervisor's profile information"""
    try:
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor()
        
        cur.execute("""
            UPDATE supervisor_profiles
            SET 
                research_interests = %s,
                department = %s,
                expertise = %s,
                preferred_projects = %s,
                office_hours = %s,
                contact_preferences = %s,
                website_url = %s,
                bio = %s
            WHERE user_id = %s
        """, (
            profile_data['research_interests'],
            profile_data['department'],
            profile_data['expertise'],
            profile_data['preferred_projects'],
            profile_data['office_hours'],
            profile_data['contact_preferences'],
            profile_data['website_url'],
            profile_data['bio'],
            supervisor_id
        ))
        
        conn.commit()
        return True
        
    except Exception as e:
        st.error(f"Error updating profile: {e}")
        return False
    finally:
        if conn:
            cur.close()
            conn.close()

def add_publication(supervisor_id, pub_data, db_config):
    """Add a new publication with duplicate checking"""
    try:
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor()
        
        # Check if publication already exists
        cur.execute("""
            SELECT id FROM supervisor_publications
            WHERE supervisor_id = %s AND title = %s AND year = %s
        """, (
            supervisor_id,
            pub_data['title'],
            pub_data['year']
        ))
        
        existing_pub = cur.fetchone()
        if existing_pub:
            return False
        
        cur.execute("""
            INSERT INTO supervisor_publications
            (supervisor_id, title, authors, year, publication_type, venue, doi, description)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            supervisor_id,
            pub_data['title'],
            pub_data['authors'],
            pub_data['year'],
            pub_data['type'],
            pub_data['venue'],
            pub_data['doi'],
            pub_data['description']
        ))
        
        conn.commit()
        return True
        
    except Exception as e:
        st.error(f"Error adding publication: {e}")
        return False
    finally:
        if conn:
            cur.close()
            conn.close()

def add_supervised_project(supervisor_id, project_data, db_config):
    """Add a supervised project with duplicate checking"""
    try:
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor()
        
        # First check if this project already exists for this supervisor
        cur.execute("""
            SELECT id FROM supervised_projects
            WHERE supervisor_id = %s AND title = %s AND year = %s AND student_name = %s
        """, (
            supervisor_id,
            project_data['title'],
            project_data['year'],
            project_data['student_name']
        ))
        
        existing_project = cur.fetchone()
        if existing_project:
            return False
        
        # If no duplicate, insert the new project
        cur.execute("""
            INSERT INTO supervised_projects
            (supervisor_id, title, student_name, year, project_type, description, outcome)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (
            supervisor_id,
            project_data['title'],
            project_data['student_name'],
            project_data['year'],
            project_data['type'],
            project_data['description'],
            project_data['outcome']
        ))
        
        conn.commit()
        return True
        
    except Exception as e:
        st.error(f"Error adding project: {e}")
        return False
    finally:
        if conn:
            cur.close()
            conn.close()

def render_profile_page(db_config):
    """Render the supervisor profile management page"""
    if not st.session_state.get('authenticated') or st.session_state.user_type != 'supervisor':
        st.error("Please login as a supervisor to access this page")
        return
    
    st.title("Profile Management")
    
    # Get current profile data
    profile_data = get_supervisor_profile(st.session_state.user['id'], db_config)
    if not profile_data:
        st.error("Error loading profile data")
        return
        
    # Safety check to ensure profile data has expected structure
    if 'profile' not in profile_data or profile_data['profile'] is None:
        st.error("Profile data format is incorrect or missing")
        return
    
    tab1, tab2, tab3 = st.tabs(["Basic Information", "Publications", "Supervised Projects"])
    
    with tab1:
        st.subheader("Profile Information")
        
        col1, col2 = st.columns(2)
        with col1:
            department = st.text_input(
                "Department", 
                value=profile_data['profile'].get('department', '')
            )
            website = st.text_input(
                "Website URL", 
                value=profile_data['profile'].get('website_url', '')
            )
        
        with col2:
            office_hours = st.text_input(
                "Office Hours", 
                value=profile_data['profile'].get('office_hours', '')
            )
            contact_prefs = st.text_input(
                "Contact Preferences",
                value=profile_data['profile'].get('contact_preferences', '')
            )
        
        research_interests = st.text_area(
            "Research Interests",
            value=profile_data['profile'].get('research_interests', ''),
            height=150
        )
        
        bio = st.text_area(
            "Professional Bio",
            value=profile_data['profile'].get('bio', ''),
            height=150
        )
        
        col1, col2 = st.columns(2)
        with col1:
            # Get existing expertise values from profile
            existing_expertise = profile_data['profile'].get('expertise', []) or []
            
            # Make sure existing values are all strings
            existing_expertise = [str(item) for item in existing_expertise]
            
            # Define comprehensive list of standard expertise options
            standard_expertise = [
                "Machine Learning", "Deep Learning", "Computer Vision", "NLP",
                "Data Science", "Cybersecurity", "Software Engineering",
                "Industrial IoT", "Robotics", "Cloud Computing", "Sentiment Analysis", 
                "Big Data Processing", "Text Analysis", "Network Programming", 
                "Algorithm Design", "Software Engineering and Distributed Computing", 
                "AI for Healthcare", "Medical Imaging", "Human-Computer-Interaction", 
                "Computational Intelligence", "Health Informatics", "Clinical Machine Learning",
                "Generative artificial intelligence", "Machine learning for code analysis",
                "Quantum Computing", "AI in Finance", "Pattern Recognition", "Transformers"
            ]
            
            # Combine both lists and ensure existing values are in options
            all_expertise_options = sorted(list(set(standard_expertise + existing_expertise)))
            
            # Debug print to console
            print(f"Existing expertise: {existing_expertise}")
            print(f"Available options: {all_expertise_options}")
            
            expertise = st.multiselect(
                "Areas of Expertise",
                options=all_expertise_options,
                default=existing_expertise
            )
        
        with col2:
            # Get existing project types from profile
            existing_projects = profile_data['profile'].get('preferred_projects', []) or []
            
            # Make sure existing values are all strings
            existing_projects = [str(item) for item in existing_projects]
            
            # Define standard project type options
            standard_projects = [
                "Research-Based", "Research-based", "Theoretical", "Theoretical Research & Analysis",
                "Industry-focused", "Software Development", "Hardware/IoT"
            ]
            
            # Combine both lists and ensure existing values are in options
            all_project_options = sorted(list(set(standard_projects + existing_projects)))
            
            # Debug print to console
            print(f"Existing projects: {existing_projects}")
            print(f"Available options: {all_project_options}")
            
            preferred_projects = st.multiselect(
                "Preferred Project Types",
                options=all_project_options,
                default=existing_projects
            )
        
        if st.button("Update Profile", type="primary"):
            updated_data = {
                'department': department,
                'research_interests': research_interests,
                'expertise': expertise,
                'preferred_projects': preferred_projects,
                'office_hours': office_hours,
                'contact_preferences': contact_prefs,
                'website_url': website,
                'bio': bio
            }
            
            if update_supervisor_profile(st.session_state.user['id'], updated_data, db_config):
                st.success("Profile updated successfully!")
                st.rerun()
    
    with tab2:
        st.subheader("Publications")
        
        with st.expander("Add New Publication"):
            title = st.text_input("Publication Title")
            
            col1, col2 = st.columns(2)
            with col1:
                authors = st.text_input("Authors (comma-separated)")
                year = st.number_input("Year", min_value=1900, max_value=datetime.now().year)
            
            with col2:
                pub_type = st.selectbox(
                    "Publication Type",
                    ["Journal Article", "Conference Paper", "Book Chapter", "Workshop Paper"]
                )
                venue = st.text_input("Venue/Journal")
            
            doi = st.text_input("DOI (optional)")
            description = st.text_area("Brief Description")
            
            if st.button("Add Publication"):
                pub_data = {
                    'title': title,
                    'authors': authors.split(','),
                    'year': year,
                    'type': pub_type,
                    'venue': venue,
                    'doi': doi,
                    'description': description
                }
                
                if add_publication(st.session_state.user['id'], pub_data, db_config):
                    st.success("Publication added successfully!")
                    st.rerun()
        
        # Display existing publications
        if profile_data['publications']:
            for pub in profile_data['publications']:
                with st.expander(f"{pub['year']} - {pub['title']}"):
                    st.write(f"**Authors:** {', '.join(pub['authors'])}")
                    st.write(f"**Type:** {pub['publication_type']}")
                    st.write(f"**Venue:** {pub['venue']}")
                    if pub['doi']:
                        st.write(f"**DOI:** {pub['doi']}")
                    if pub['description']:
                        st.write(f"**Description:** {pub['description']}")
    
    with tab3:
        st.subheader("Supervised Projects")
        
        with st.expander("Add Past Project"):
            title = st.text_input("Project Title", key="project_title")
            
            col1, col2 = st.columns(2)
            with col1:
                student_name = st.text_input("Student Name")
                year = st.number_input("Year", 
                                     min_value=1900, 
                                     max_value=datetime.now().year,
                                     key="project_year")
            
            with col2:
                project_type = st.selectbox(
                    "Project Type",
                    ["Undergraduate", "Masters", "PhD", "Research Assistant"]
                )
            
            description = st.text_area("Project Description", key="project_desc")
            outcome = st.text_area("Project Outcome")
            
            if st.button("Add Project"):
                project_data = {
                    'title': title,
                    'student_name': student_name,
                    'year': year,
                    'type': project_type,
                    'description': description,
                    'outcome': outcome
                }
                
                if add_supervised_project(st.session_state.user['id'], project_data, db_config):
                    st.success("Project added successfully!")
                    st.rerun()
        
        # Display existing projects
        if profile_data['projects']:
            for project in profile_data['projects']:
                with st.expander(f"{project['year']} - {project['title']}"):
                    st.write(f"**Student:** {project['student_name']}")
                    st.write(f"**Type:** {project['project_type']}")
                    st.write(f"**Description:** {project['description']}")
                    if project['outcome']:
                        st.write(f"**Outcome:** {project['outcome']}")

if __name__ == "__main__":
    from auth_app import DB_CONFIG
    render_profile_page(DB_CONFIG)
