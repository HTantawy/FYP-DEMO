import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, timedelta
from auth_app import DB_CONFIG
from messaging import display_messages_tab

def get_supervisor_requests(supervisor_id):
    """Fetch all requests for a supervisor"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        # Add debugging to check supervisor_id value
        print(f"Getting requests for supervisor ID: {supervisor_id}")
        
        cur.execute("""
            SELECT 
                sr.id as request_id,
                sr.project_title,
                sr.project_description,
                sr.status,
                sr.matching_score,
                sr.created_at,
                sr.student_id,
                u.full_name as student_name,
                u.email as student_email,
                sp.course,
                sp.year_of_study
            FROM supervisor_requests sr
            JOIN users u ON sr.student_id = u.id
            JOIN student_profiles sp ON u.id = sp.user_id
            WHERE sr.supervisor_id = %s
            ORDER BY CASE 
                WHEN sr.status = 'pending' THEN 1
                WHEN sr.status = 'accepted' THEN 2
                ELSE 3
            END, sr.created_at DESC
        """, (supervisor_id,))
        
        requests = cur.fetchall()
        
        # Debugging - print how many requests were found
        print(f"Found {len(requests)} requests for supervisor ID {supervisor_id}")
        
        return requests
        
    except Exception as e:
        st.error(f"Error fetching requests: {e}")
        print(f"Database error in get_supervisor_requests: {e}")
        return []
        
    finally:
        if conn:
            cur.close()
            conn.close()

def update_request_status(request_id, new_status):
    """Update the status of a request"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        cur.execute("""
            UPDATE supervisor_requests
            SET status = %s, updated_at = NOW()
            WHERE id = %s
            RETURNING student_id
        """, (new_status, request_id))
        
        student_id = cur.fetchone()[0]
        
        # Add notification for the student
        cur.execute("""
            INSERT INTO notifications (user_id, message, type)
            VALUES (%s, %s, 'request_update')
        """, (
            student_id,
            f"Your supervision request has been {new_status}"
        ))
        
        conn.commit()
        return True
        
    except Exception as e:
        st.error(f"Error updating request: {e}")
        print(f"Database error in update_request_status: {e}")
        return False
        
    finally:
        if conn:
            cur.close()
            conn.close()

def get_request_statistics(supervisor_id):
    """Get statistics about supervisor requests"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        # Debug the supervisor ID being used
        print(f"Getting statistics for supervisor ID: {supervisor_id}")
        
        # Get total count of requests for this supervisor
        cur.execute("""
            SELECT COUNT(*) as total
            FROM supervisor_requests
            WHERE supervisor_id = %s
        """, (supervisor_id,))
        total_count = cur.fetchone()['total']
        print(f"Total count from database: {total_count}")
        
        # Get status counts - ensure filtering by supervisor_id
        cur.execute("""
            SELECT status, COUNT(*) as count
            FROM supervisor_requests
            WHERE supervisor_id = %s
            GROUP BY status
        """, (supervisor_id,))
        status_counts = cur.fetchall()
        print(f"Status counts: {status_counts}")
        
        # Get weekly request counts - ensure filtering by supervisor_id
        cur.execute("""
            SELECT DATE_TRUNC('week', created_at) as week, COUNT(*) as count
            FROM supervisor_requests
            WHERE supervisor_id = %s
            AND created_at > NOW() - INTERVAL '6 months'
            GROUP BY week
            ORDER BY week
        """, (supervisor_id,))
        weekly_counts = cur.fetchall()
        
        # Verify counts add up correctly
        total_from_status = sum(item['count'] for item in status_counts) if status_counts else 0
        print(f"Total from status counts: {total_from_status}")
        
        # Create simplified return dictionary to ensure consistency
        result = {
            'total_count': total_count,
            'status_counts': status_counts,
            'weekly_counts': weekly_counts
        }
        
        return result
        
    except Exception as e:
        st.error(f"Error fetching statistics: {e}")
        print(f"Database error in get_request_statistics: {e}")
        return None
        
    finally:
        if conn:
            cur.close()
            conn.close()

def create_statistics_charts(stats):
    """Create statistics visualizations"""
    if not stats or not stats['status_counts']:
        return None
        
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Request Status Distribution", "Weekly Requests"),
        specs=[[{"type": "pie"}, {"type": "scatter"}]]
    )
    
    # Status distribution pie chart
    labels = [item['status'] for item in stats['status_counts']]
    values = [item['count'] for item in stats['status_counts']]
    
    # Debug color mapping
    color_map = {'pending': '#FFA500', 'accepted': '#4CAF50', 'rejected': '#F44336'}
    colors = [color_map.get(label, '#CCCCCC') for label in labels]
    
    fig.add_trace(
        go.Pie(labels=labels, values=values, 
               textinfo='label+percent',
               marker=dict(colors=colors)),
        row=1, col=1
    )
    
    # Weekly requests line chart
    if stats['weekly_counts']:
        weeks = [item['week'].strftime('%Y-%m-%d') for item in stats['weekly_counts']]
        counts = [item['count'] for item in stats['weekly_counts']]
        fig.add_trace(
            go.Scatter(x=weeks, y=counts, mode='lines+markers',
                       name='Weekly Requests',
                       line=dict(color='#2196F3')),
            row=1, col=2
        )
    
    fig.update_layout(
        height=400,
        showlegend=True,
        template='plotly_white'
    )
    return fig

def show_requests_section(requests):
    """Display requests with advanced filtering and interactive table"""
    st.subheader("Student Requests")
    
    # We keep the DataFrame styling but remove the extra .filter-container
    st.markdown("""
        <style>
        .stDataFrame {
            background-color: white;
            padding: 1rem;
            border-radius: 0.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Filter UI: simply place columns without the extra HTML container
    col1, col2, col3 = st.columns([2,1,1])
    
    with col1:
        search_query = st.text_input("🔍 Search by student name or project title")
    with col2:
        status_filter = st.selectbox(
            "Status",
            ["All", "Pending", "Accepted", "Rejected"]
        )
    with col3:
        if requests and len(requests) > 0:
            years = sorted(list(set(r['year_of_study'] for r in requests)))
            year_filter = st.selectbox("Year of Study", ["All"] + years)
        else:
            year_filter = "All"
        
    col4, col5 = st.columns(2)
    with col4:
        if requests and len(requests) > 0:
            courses = sorted(list(set(r['course'] for r in requests)))
            course_filter = st.selectbox("Course", ["All"] + courses)
        else:
            course_filter = "All"
    with col5:
        date_range = st.selectbox(
            "Time Period",
            ["All Time", "Last Week", "Last Month", "Last 3 Months", "Custom"]
        )
        
    # Show custom date range if selected
    if date_range == "Custom":
        start_date, end_date = st.date_input(
            "Select Date Range",
            value=(datetime.now() - timedelta(days=30), datetime.now()),
            max_value=datetime.now()
        )

    # Apply filters - make sure we have requests to filter
    filtered_requests = requests.copy() if requests else []
    
    # Search filter
    if search_query and filtered_requests:
        filtered_requests = [
            r for r in filtered_requests
            if search_query.lower() in r['student_name'].lower() 
               or search_query.lower() in r['project_title'].lower()
        ]
    
    # Status filter
    if status_filter != "All" and filtered_requests:
        filtered_requests = [r for r in filtered_requests if r['status'].lower() == status_filter.lower()]
        
    # Year filter
    if year_filter != "All" and filtered_requests:
        filtered_requests = [r for r in filtered_requests if r['year_of_study'] == year_filter]
        
    # Course filter
    if course_filter != "All" and filtered_requests:
        filtered_requests = [r for r in filtered_requests if r['course'] == course_filter]
        
    # Date filter
    if date_range != "All Time" and filtered_requests:
        current_date = datetime.now()
        if date_range == "Last Week":
            start_date = current_date - timedelta(days=7)
        elif date_range == "Last Month":
            start_date = current_date - timedelta(days=30)
        elif date_range == "Last 3 Months":
            start_date = current_date - timedelta(days=90)
        elif date_range == "Custom":
            start_date = datetime.combine(start_date, datetime.min.time())
            current_date = datetime.combine(end_date, datetime.max.time())
        
        filtered_requests = [
            r for r in filtered_requests 
            if start_date <= r['created_at'] <= current_date
        ]

    # Show results count
    st.write(f"Showing {len(filtered_requests)} requests")
    
    # Convert to DataFrame for interactive table
    if filtered_requests:
        df = pd.DataFrame(filtered_requests)
        df['created_at'] = pd.to_datetime(df['created_at']).dt.strftime('%Y-%m-%d %H:%M')
        df['matching_score'] = df['matching_score'].round(2)
        
        # Select and rename columns for display
        display_df = df[[
            'student_name', 'course', 'year_of_study', 
            'project_title', 'matching_score', 'status', 'created_at'
        ]].rename(columns={
            'student_name': 'Student',
            'course': 'Course',
            'year_of_study': 'Year',
            'project_title': 'Project',
            'matching_score': 'Match Score',
            'status': 'Status',
            'created_at': 'Submitted'
        })
        
        # Create interactive table
        st.dataframe(
            display_df,
            use_container_width=True,
            height=400,
            hide_index=True,
            column_config={
                "Match Score": st.column_config.NumberColumn(
                    format="%.2f",
                    help="Matching score between student and supervisor"
                ),
                "Status": st.column_config.SelectboxColumn(
                    options=["pending", "accepted", "rejected"],
                    required=True
                )
            }
        )
        
        # Export functionality
        if st.button("Export Filtered Data"):
            csv = display_df.to_csv(index=False)
            st.download_button(
                label="Download as CSV",
                data=csv,
                file_name=f"supervision_requests_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        # Show detailed view for selected requests
        st.subheader("Detailed View")
        for request in filtered_requests:
            with st.expander(f"{request['project_title']} - {request['student_name']}", expanded=False):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write("**Student Details:**")
                    st.write(f"Name: {request['student_name']}")
                    st.write(f"Email: {request['student_email']}")
                    st.write(f"Course: {request['course']} (Year {request['year_of_study']})")
                    
                    st.write("\n**Project Details:**")
                    st.write(request['project_description'])
                    
                with col2:
                    status_colors = {
                        'pending': 'orange',
                        'accepted': 'green',
                        'rejected': 'red'
                    }
                    st.markdown(
                        f"**Status:** :{status_colors[request['status']]}[{request['status'].upper()}]"
                    )
                    
                    st.write(f"**Match Score:** {request['matching_score']:.2f}")
                    st.write(f"**Submitted:** {request['created_at']}")
                    
                    if request['status'] == 'pending':
                        if st.button("Accept", key=f"accept_{request['request_id']}"):
                            if update_request_status(request['request_id'], 'accepted'):
                                st.success("Request accepted!")
                                st.rerun()
                        
                        if st.button("Reject", key=f"reject_{request['request_id']}"):
                            if update_request_status(request['request_id'], 'rejected'):
                                st.success("Request rejected!")
                                st.rerun()
    else:
        st.info("No requests match the selected filters.")

def supervisor_dashboard():
    """Main supervisor dashboard page"""
    if not st.session_state.get('authenticated') or st.session_state.user_type != 'supervisor':
        st.error("Please login as a supervisor to access this page")
        return
    
    # Debug information about the authenticated user
    supervisor_id = st.session_state.user['id']
    print(f"Supervisor Dashboard - User ID: {supervisor_id}, Type: {st.session_state.user_type}")
    
    # Instead of st.title("Supervisor Dashboard"), we use a custom wrapper:
    st.markdown("""
        <div style="background-color: #fff; padding: 1rem; border-radius: 8px; 
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 1rem;">
            <h1 style="font-size: 3.5rem; font-weight: 600; color: #1E3D59; margin: 0;">
                Supervisor Dashboard
            </h1>
        </div>
    """, unsafe_allow_html=True)
    
    # Create tabs (this approach avoids extra blank bars)
    tab_overview, tab_requests, tab_messages,tab_profile = st.tabs(["Overview", "Requests", "Messages", "Profile"])
    
    # Overview Tab
    with tab_overview:
        st.subheader("Overview")
        
        # Inject some custom CSS for the KPI cards
        st.markdown("""
            <style>
            .card-container {
                display: flex;
                justify-content: space-around;
                margin-top: 1.5rem;
                margin-bottom: 1.5rem;
            }
            .card {
                background-color: #ffffff;
                border-radius: 0.5rem;
                padding: 1.5rem;
                text-align: center;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                width: 30%;
                transition: all 0.3s ease-in-out;
                border: 2px solid transparent;       
            }
            .card:hover {
                box-shadow: 0 6px 10px rgba(0,0,0,0.15); /* Slightly bigger shadow on hover */
                transform: translateY(-5px); /* Lift the card up */
                border-color: #4F46E5;    
            }        
            .card-value {
                font-size: 2rem;
                font-weight: bold;
                margin-bottom: 0.3rem;
                color: #4F46E5
            }
            .card-label {
                font-size: 1.1rem;
                color: #666;
            }
            </style>
        """, unsafe_allow_html=True)

        # Get and display statistics
        stats = get_request_statistics(supervisor_id)
        if stats:
            fig = create_statistics_charts(stats)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Calculate summary metrics - fixed implementation
            total_requests = stats['total_count']
            
            # Extract pending requests count
            pending_requests = 0
            for item in stats['status_counts']:
                if item['status'] == 'pending':
                    pending_requests = item['count']
                    break
            
            # Extract accepted requests count
            accepted_count = 0
            for item in stats['status_counts']:
                if item['status'] == 'accepted':
                    accepted_count = item['count']
                    break
            
            # Calculate acceptance rate
            acceptance_rate = (accepted_count / total_requests * 100) if total_requests > 0 else 0

            # Display the KPI cards
            st.markdown(f"""
                <div class="card-container">
                    <div class="card">
                        <div class="card-value">{total_requests}</div>
                        <div class="card-label">Total Requests</div>
                    </div>
                    <div class="card">
                        <div class="card-value">{pending_requests}</div>
                        <div class="card-label">Pending Requests</div>
                    </div>
                    <div class="card">
                        <div class="card-value">{acceptance_rate:.1f}%</div>
                        <div class="card-label">Acceptance Rate</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
    
    # Requests Tab
    with tab_requests:
        # Get requests specific to this supervisor
        requests = get_supervisor_requests(supervisor_id)
        show_requests_section(requests)

    with tab_messages:
        display_messages_tab(DB_CONFIG)    
    
    # Profile Tab
    with tab_profile:
        from supervisor_profile import render_profile_page
        render_profile_page(DB_CONFIG)
    
    # Sidebar with custom CSS for the logout button
    with st.sidebar:
        st.markdown("""
            <style>
            [data-testid="stSidebar"] .stButton > button {
                background-color: #4F46E5;
                color: white;
                border: none;
                border-radius: 8px;
            }
            </style>
        """, unsafe_allow_html=True)
        st.write(f"Welcome, {st.session_state.user['full_name']}")
        if st.button("Logout", key="supervisor_logout"):
            st.session_state.clear()
            st.rerun()

if __name__ == "__main__":
    supervisor_dashboard()


        
        
       