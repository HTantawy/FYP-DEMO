# supervisor_view.py

import streamlit as st
import psycopg2
from psycopg2.extras import RealDictCursor
from supervisor_profile import get_supervisor_profile

def view_supervisor_profile(supervisor_id, supervisor_name, db_config):
    """Display a read-only view of a supervisor's profile"""
    
    
    profile_data = get_supervisor_profile(supervisor_id, db_config)
    if not profile_data:
        st.error("Error loading profile data")
        return
        
    
    st.title(f"Prof. {supervisor_name}'s Profile")
    if profile_data['profile'].get('department'):
        st.subheader(f"Department of {profile_data['profile']['department']}")
    
    
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            if profile_data['profile'].get('office_hours'):
                st.info(f"🕒 Office Hours: {profile_data['profile']['office_hours']}")
            if profile_data['profile'].get('contact_preferences'):
                st.info(f"📧 Contact Preferences: {profile_data['profile']['contact_preferences']}")
        with col2:
            if profile_data['profile'].get('website_url'):
                st.markdown(f"🔗 [Personal Website]({profile_data['profile']['website_url']})")
    
    
    tab1, tab2, tab3 = st.tabs(["Research Profile", "Publications", "Past Projects"])
    
    
    with tab1:
        if profile_data['profile'].get('bio'):
            st.markdown("### About")
            st.write(profile_data['profile']['bio'])
        
        st.markdown("### Research Interests")
        st.write(profile_data['profile'].get('research_interests', 'Not specified'))
        
        col1, col2 = st.columns(2)
        with col1:
            if profile_data['profile'].get('expertise'):
                st.markdown("### Areas of Expertise")
                for area in profile_data['profile']['expertise']:
                    st.markdown(f"- {area}")
        
        with col2:
            if profile_data['profile'].get('preferred_projects'):
                st.markdown("### Preferred Project Types")
                for project_type in profile_data['profile']['preferred_projects']:
                    st.markdown(f"- {project_type}")
    
    
    with tab2:
        if profile_data['publications']:
            
            publications_by_year = {}
            for pub in profile_data['publications']:
                year = pub['year']
                if year not in publications_by_year:
                    publications_by_year[year] = []
                publications_by_year[year].append(pub)
            
            # Display publications by year
            for year in sorted(publications_by_year.keys(), reverse=True):
                st.markdown(f"### {year}")
                for pub in publications_by_year[year]:
                    with st.expander(pub['title']):
                        st.write(f"**Authors:** {', '.join(pub['authors'])}")
                        st.write(f"**Venue:** {pub['venue']}")
                        if pub['doi']:
                            st.write(f"**DOI:** [{pub['doi']}](https://doi.org/{pub['doi']})")
                        if pub['description']:
                            st.write(f"**Abstract:** {pub['description']}")
        else:
            st.info("No publications listed")
    
    # Past Projects Tab
    with tab3:
        if profile_data['projects']:
            
            projects_by_type = {}
            for project in profile_data['projects']:
                proj_type = project['project_type']
                if proj_type not in projects_by_type:
                    projects_by_type[proj_type] = []
                projects_by_type[proj_type].append(project)
            
            
            for proj_type in projects_by_type:
                st.markdown(f"### {proj_type} Projects")
                for project in sorted(projects_by_type[proj_type], 
                                    key=lambda x: x['year'], 
                                    reverse=True):
                    with st.expander(f"{project['year']} - {project['title']}"):
                        st.write(f"**Student:** {project['student_name']}")
                        st.write(f"**Description:** {project['description']}")
                        if project['outcome']:
                            st.write(f"**Outcome:** {project['outcome']}")
        else:
            st.info("No past projects listed")