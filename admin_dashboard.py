import streamlit as st
import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
from auth_app import DB_CONFIG
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import io

def admin_dashboard():
    if not st.session_state.get('authenticated') or st.session_state.user_type != 'admin':
        st.error("Please login as an admin to access this page")
        return
    
    # --- Custom CSS for better styling ---
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: 600;
            color: #1E3D59;
            margin-bottom: 2rem;
        }
        .stat-card {
            background-color: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
            margin-bottom: 1rem;
            border: 2px solid transparent; /* Add this line for transparent border */
            transition: all 0.3s ease-in-out;    
        }
        .stat-card:hover {
            box-shadow: 0 6px 10px rgba(0,0,0,0.15); /* Bigger shadow on hover */
            transform: translateY(-5px); /* Lift the card up slightly */
            border-color: #4F46E5; /* Change border color to match the app's color scheme */
        }        
        .stat-number {
            font-size: 2rem;
            font-weight: bold;
            color: #2E5073;
        }
        .stat-label {
            font-size: 1rem;
            color: #666;
            margin-top: 0.5rem;
        }
        .data-table {
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .section-header {
            font-size: 1.5rem;
            color: #1E3D59;
            margin: 1.5rem 0;
            font-weight: 500;
        }
        .status-badge {
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 500;
        }
        .status-badge-allocated {
            background-color: #d4edda;
            color: #155724;
        }
        .status-badge-unallocated {
            background-color: #f8d7da;
            color: #721c24;
        }
        .stButton > button {
            border-radius: 8px;
            font-weight: 500;
            background-color: #4F46E5;  
            color: white !important;
        }
        .stButton > button:hover {
        background-color: #4338CA !important;
    }        
         .stDownloadButton button {
            border-radius: 8px;
            font-weight: 500;
            background-color: #4F46E5 !important;  
            color: white !important;
        }
        .stDownloadButton button:hover {
        background-color: #4338CA !important;
        }               
        div[data-testid="stExpander"] {
            border-radius: 8px !important;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1) !important;
            margin-bottom: 1rem !important;
            background-color: #ffffff !important;
            border: 1px solid #e1e1e1 !important;
        }
        div[data-testid="stExpander"] > div {
            padding: 0rem !important;
        }
        div[data-testid="stExpander"] .st-expanderHeader {
            margin: 0 !important;
            padding: 0.5rem 1rem !important;
        }
        
        /* Student card body inside the expander */
        .student-card-body {
            padding: 1rem;
        }
        .student-card-body p {
            margin: 0.3rem 0;
        }
        
        .search-box {
            background-color: white;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        /* For card-based layout in Student Allocations tab */
        .allocation-cards-container {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 1.5rem;
            margin-top: 1rem;
            width: 100%;
        }
        .allocation-card {
            background-color: #fff;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            padding: 1rem;
            position: relative;
            height: fit-content;
            min-width: 0;
            border: 2px solid transparent;
            transition: all 0.3s ease-in-out;        
        }
        .allocation-card:hover {
            box-shadow: 0 6px 10px rgba(0,0,0,0.15); /* Bigger shadow on hover */
            transform: translateY(-5px); /* Lift the card up slightly */
            border-color: #4F46E5; /* Change border color to match the app's color scheme */
        }       
        .allocation-card h4 {
            margin: 0;
            font-size: 1.1rem;
            font-weight: 600;
            color: #1E3D59;
            margin-bottom: 0.3rem;
        }
        .allocation-card p {
            margin: 0.25rem 0;
            font-size: 0.95rem;
        }
        .allocation-badge {
            position: absolute;
            top: 1rem;
            right: 1rem;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.85rem;
            font-weight: 600;
        }
        .allocation-badge-allocated {
            background-color: #d4edda;
            color: #155724;
        }
        .allocation-badge-unallocated {
            background-color: #f8d7da;
            color: #721c24;
        }

        /* Scrollbars for dataframes */
        .stDataFrame {
            background-color: white !important;
            border-radius: 8px !important;
            overflow: hidden !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
        }
        .stDataFrame table {
            border-collapse: collapse !important;
            width: 100% !important;
        }
        .stDataFrame thead tr th {
            background-color: #1E3D59 !important;
            color: white !important;
            padding: 12px 16px !important;
            font-weight: 600 !important;
            text-align: left !important;
            border: none !important;
        }
        .stDataFrame tbody tr {
            border: none !important;
            transition: background-color 0.2s ease !important;
        }
        .stDataFrame tbody tr:nth-child(even) {
            background-color: #f8f9fa !important;
        }
        .stDataFrame tbody tr:hover {
            background-color: #f0f4f8 !important;
        }
        .stDataFrame td {
            padding: 12px 16px !important;
            border: none !important;
            font-size: 14px !important;
            color: #1E3D59 !important;
        }
        .year-badge {
            background-color: #f1f3f4;
            padding: 4px 8px;
            border-radius: 4px;
            font-weight: 500;
            color: #3c4043;
        }
        .course-badge {
            color: #1a73e8;
            font-weight: 500;
        }
        .supervisor-badge {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .score-cell {
            padding: 6px 12px !important;
            border-radius: 12px !important;
            font-weight: 500 !important;
            text-align: center !important;
            display: inline-block !important;
            min-width: 60px !important;
        }
        .stDataFrame td, .stDataFrame th {
            border: none !important;
        }
        .stDataFrame div::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        .stDataFrame div::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 4px;
        }
        .stDataFrame div::-webkit-scrollbar-thumb {
            background: #c1c1c1;
            border-radius: 4px;
        }
        .stDataFrame div::-webkit-scrollbar-thumb:hover {
            background: #a8a8a8;
        }

        /* Extra styling for the new card-based matches overview */
        .match-card {
            background-color: #ffffff;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .match-card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.5rem;
        }
        .match-card-header h4 {
            margin: 0;
            font-size: 1.15rem;
            color: #1E3D59;
        }
        .match-card-header .match-id {
            color: #5f6368;
            font-size: 0.85rem;
        }
        .match-card-body p {
            margin: 0.3rem 0;
        }
        .score-badge {
            display: inline-block;
            padding: 6px 12px;
            border-radius: 12px;
            font-weight: 600;
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Sidebar with improved styling
    with st.sidebar:
        st.markdown("""
            <div style='padding: 1rem;'>
                <h4 style='color: #1E3D59;'>Admin Controls</h4>
            </div>
        """, unsafe_allow_html=True)
        if st.button("← Back to Login", use_container_width=True):
            st.session_state.clear()
            st.rerun()

    # Main header
    st.markdown("""
        <div style="background-color: #fff; padding: 1rem; border-radius: 8px; 
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 1rem;">
            <h1 style="font-size: 3.5rem; font-weight: 600; color: #1E3D59; margin: 0;">
                Admin Dashboard
            </h1>
        </div>
    """, unsafe_allow_html=True)    

    # Enhanced tabs
    tabs = st.tabs(["📊 Matches Overview", "👥 Manage Supervisors and Students", "📋 Student Allocations"])

    with tabs[0]:
        show_matches_overview()
    with tabs[1]:
        manage_supervisors_students()
    with tabs[2]:
        show_student_allocations()


def show_matches_overview():
    """Display each match in a more card-like, user-friendly format."""
    st.markdown('<h2 class="section-header">Matches Overview</h2>', unsafe_allow_html=True)
    matches = get_all_matches()

    if matches.empty:
        st.info("No matches found.")
        return
    
    # Loop through each match and present it as a 'card'
    for idx, row in matches.iterrows():
        # Decide on the color styles for the final_score
        score_style = get_score_style(row['final_score'])
        created_str = row['created_at'].strftime('%Y-%m-%d %H:%M')

        st.markdown(f"""
            <div class="match-card">
                <div class="match-card-header">
                    <h4> Match Summary</h4>
                    <span class="match-id">ID #{row['id']}</span>
                </div>
                <div class="match-card-body">
                    <p><strong>Student:</strong> {row['student_name']}</p>
                    <p><strong>Supervisor:</strong> {row['supervisor_name']}</p>
                    <p><strong>Final Score:</strong>
                        <span class="score-badge" style="background-color:{score_style['bg']}; color:{score_style['color']};">
                            {row['final_score']:.2f}
                        </span>
                    </p>
                    <p style="color:#5f6368; font-size:0.9rem; margin-top:0.75rem;">
                        <em>Created At: {created_str}</em>
                    </p>
                </div>
            </div>
        """, unsafe_allow_html=True)


def get_score_style(score):
    """Return background and text color for the score badge based on thresholds."""
    if score >= 0.8:
        return {'bg': '#d4edda', 'color': '#155724'}
    elif score >= 0.6:
        return {'bg': '#c8e6c9', 'color': '#2E7D32'}
    elif score >= 0.4:
        return {'bg': '#fff3cd', 'color': '#856404'}
    else:
        return {'bg': '#f8d7da', 'color': '#721c24'}


def show_student_allocations():
    st.markdown('<h2 class="section-header">Student Allocations by Hussein tantawy</h2>', unsafe_allow_html=True)
    
    students = get_all_students_with_allocation()
    if students.empty:
        st.info("No students found.")
        return
        
    # Enhanced filters section
    st.markdown('<div class="search-box">', unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    with col1:
        search_term = st.text_input("🔍 Search by name or course...")
    with col2:
        status_filter = st.selectbox("All Students", ["All", "Allocated", "Unallocated"], index=0)
    st.markdown('</div>', unsafe_allow_html=True)

    # Apply filters
    filtered_students = students
    if status_filter != "All":
        filtered_students = filtered_students[filtered_students['allocation_status'] == status_filter.lower()]
    if search_term:
        mask = (
            filtered_students['full_name'].str.contains(search_term, case=False, na=False) |
            filtered_students['course'].str.contains(search_term, case=False, na=False)
        )
        filtered_students = filtered_students[mask]
    
    # Statistics cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{len(students)}</div>
                <div class="stat-label">Total Students</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        allocated = len(students[students['allocation_status'] == 'allocated'])
        st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{allocated}</div>
                <div class="stat-label">Allocated Students</div>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        unallocated = len(students[students['allocation_status'] == 'unallocated'])
        st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{unallocated}</div>
                <div class="stat-label">Unallocated Students</div>
            </div>
        """, unsafe_allow_html=True)
    
    # --- Now, show results as containers/cards (instead of a DataFrame) ---
    st.markdown('<div class="allocation-cards-container">', unsafe_allow_html=True)
    for _, row in filtered_students.iterrows():
        status_class = "allocation-badge-allocated" if row['allocation_status'] == 'allocated' else "allocation-badge-unallocated"
        status_label = row['allocation_status'].title()  # "Allocated" or "Unallocated"
        supervisor_display = row['supervisor_name'] if row['supervisor_name'] not in [None, "None"] else "-"

        st.markdown(f"""
            <div class="allocation-card">
                <h4>{row['full_name']}</h4>
                <p style="color:#5f6368;">{row['email']}</p>
                <p><strong>Course:</strong> {row['course']}</p>
                <p><strong>Year:</strong> Year {row['year_of_study']}</p>
                <p>
                  <span style="background-color:#e9ecef; color:#444; padding:4px 8px; border-radius:20px; font-size:0.85rem; font-weight:600;">
                    {supervisor_display}
                  </span>
                </p>
                <div class="allocation-badge {status_class}">{status_label}</div>
            </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Export Options: PDF and CSV ---
    col_pdf, col_csv = st.columns(2)
    with col_pdf:
        if st.button("📥 Export PDF", key="export_pdf", use_container_width=True):
            with st.spinner("Generating PDF..."):
                generate_pdf_report(filtered_students)
    with col_csv:
        csv_data = filtered_students.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export CSV",
            data=csv_data,
            file_name=f"student_allocations_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime='text/csv',
            key="download_csv",
            use_container_width=True
        )


def generate_pdf_report(students_df):
    """Generate PDF report of student allocations with improved styling"""
    try:
        # Create a buffer to store PDF
        buffer = io.BytesIO()
        
        # Create PDF document with custom styling
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=40,
            leftMargin=40,
            topMargin=40,
            bottomMargin=40
        )
        
        # Define styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.HexColor('#1E3D59'),
            alignment=1  # Center alignment
        )
        
        subtitle_style = ParagraphStyle(
            'CustomSubtitle',
            parent=styles['Normal'],
            fontSize=12,
            textColor=colors.HexColor('#666666'),
            alignment=1,  # Center alignment
            spaceAfter=20
        )
        
        # Container for elements
        elements = []
        
        # Add title and timestamp
        elements.append(Paragraph("Student Allocations Report", title_style))
        elements.append(Paragraph(
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            subtitle_style
        ))
        elements.append(Spacer(1, 20))
        
        # Add statistics
        total_students = len(students_df)
        allocated = len(students_df[students_df['allocation_status'] == 'allocated'])
        unallocated = total_students - allocated
        
        stats_style = ParagraphStyle(
            'Stats',
            parent=styles['Normal'],
            fontSize=12,
            spaceAfter=20,
            alignment=0  # Left alignment
        )
        
        stats_text = f"""
        <b>Summary Statistics:</b>
        • Total Students: {total_students}
        • Allocated Students: {allocated} ({(allocated/total_students*100 if total_students > 0 else 0):.1f}%)
        • Unallocated Students: {unallocated} ({(unallocated/total_students*100 if total_students > 0 else 0):.1f}%)
        """
        elements.append(Paragraph(stats_text, stats_style))
        elements.append(Spacer(1, 20))
        
        # Prepare table data
        headers = ['Full Name', 'Email', 'Course', 'Year', 'Status', 'Supervisor']
        data = [headers]
        
        for _, row in students_df.iterrows():
            data.append([
                row['full_name'],
                row['email'],
                row['course'],
                str(row['year_of_study']),
                row['allocation_status'].title(),
                row['supervisor_name'] if row['supervisor_name'] not in ['None', '-'] else '-'
            ])
        
        # Create and style table
        table = Table(data, repeatRows=1)
        table.setStyle(TableStyle([
            # Headers
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E3D59')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('TOPPADDING', (0, 0), (-1, 0), 12),
            
            # Table content
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
            ('TOPPADDING', (0, 1), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')])
        ]))
        
        elements.append(table)
        
        # Build PDF
        doc.build(elements)
        
        # Trigger download
        st.download_button(
            label="💾 Download PDF",
            data=buffer.getvalue(),
            file_name=f"student_allocations_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime='application/pdf',
            key='download_pdf',
            use_container_width=False
        )
        
    except Exception as e:
        st.error(f"Error generating PDF: {str(e)}")


def manage_supervisors_students():
    st.markdown('<h2 class="section-header">Manage Supervisors and Students</h2>', unsafe_allow_html=True)
    
    supervisors = get_supervisors()
    if supervisors.empty:
        st.info("No supervisors found.")
        return

    selected_supervisor = st.selectbox(
        "👨‍🏫 Select Supervisor", 
        supervisors['full_name'].tolist(),
        key="supervisor_select"
    )

    if selected_supervisor:
        supervisor_id = int(supervisors.loc[supervisors['full_name'] == selected_supervisor, 'id'].iloc[0])
        show_supervisor_details(supervisor_id, selected_supervisor)


def show_supervisor_details(supervisor_id, supervisor_name):
    students = get_students_under_supervisor(supervisor_id)
    
    st.markdown(f'<h3 class="section-header">Students under {supervisor_name}</h3>', unsafe_allow_html=True)
    
    if students.empty:
        st.info("No students assigned to this supervisor.")
        return
    
    for _, student in students.iterrows():
        with st.expander(f"📚 {student['full_name']} - {student['course']}"):
            st.markdown("<div class='student-card-body'>", unsafe_allow_html=True)
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"""
                    <p><strong>Email:</strong> {student['email']}</p>
                    <p><strong>Year of Study:</strong> {student['year_of_study']}</p>
                    <p><strong>Course:</strong> {student['course']}</p>
                """, unsafe_allow_html=True)
            
            with col2:
                if st.button("🗑️ Remove", key=f"remove_{student['id']}"):
                    if remove_student_from_supervisor(int(student['id']), supervisor_id):
                        st.success(f"Removed {student['full_name']} from {supervisor_name}")
                        st.rerun()
                
                other_supervisors = get_supervisors()
                other_supervisors = other_supervisors[other_supervisors['id'] != supervisor_id]
                
                new_supervisor = st.selectbox(
                    "Transfer to",
                    other_supervisors['full_name'].tolist(),
                    key=f"move_{student['id']}"
                )
                
                if new_supervisor and st.button("↗️ Transfer", key=f"move_btn_{student['id']}"):
                    new_supervisor_id = int(other_supervisors.loc[
                        other_supervisors['full_name'] == new_supervisor, 'id'
                    ].iloc[0])
                    
                    if move_student_to_supervisor(int(student['id']), new_supervisor_id, supervisor_id):
                        st.success(f"Moved {student['full_name']} to {new_supervisor}")
                        st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)


# -------------------------
# Database Helper Functions
# -------------------------
def get_all_students_with_allocation():
    """Fetch all students with their allocation status."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        cur.execute("""
            WITH latest_matches AS (
                SELECT DISTINCT ON (student_id) 
                    student_id,
                    supervisor_id,
                    'allocated' as allocation_status
                FROM matching_history
                ORDER BY student_id, created_at DESC
            )
            SELECT 
                u.id,
                u.full_name,
                u.email,
                sp.course,
                sp.year_of_study,
                COALESCE(lm.allocation_status, 'unallocated') as allocation_status,
                sup.full_name as supervisor_name
            FROM users u
            JOIN student_profiles sp ON u.id = sp.user_id
            LEFT JOIN latest_matches lm ON u.id = lm.student_id
            LEFT JOIN users sup ON lm.supervisor_id = sup.id
            WHERE u.user_type = 'student'
            ORDER BY u.full_name
        """)
        
        students = cur.fetchall()
        return pd.DataFrame(students)
    except Exception as e:
        st.error(f"Error fetching students: {e}")
        return pd.DataFrame()
    finally:
        if conn:
            cur.close()
            conn.close()

def get_students_under_supervisor(supervisor_id):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor(cursor_factory=RealDictCursor)

        cur.execute("""
            SELECT DISTINCT 
                u.id, 
                u.full_name, 
                u.email, 
                sp.course, 
                sp.year_of_study
            FROM matching_history mh
            JOIN users u ON mh.student_id = u.id
            JOIN student_profiles sp ON u.id = sp.user_id
            WHERE mh.supervisor_id = %s
            ORDER BY u.full_name
        """, (supervisor_id,))
        
        students = cur.fetchall()
        return pd.DataFrame(students)
    except Exception as e:
        st.error(f"Error fetching students: {e}")
        return pd.DataFrame()
    finally:
        if conn:
            cur.close()
            conn.close()

def get_supervisors():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor(cursor_factory=RealDictCursor)

        cur.execute("""
            SELECT u.id, u.full_name, u.email
            FROM users u
            WHERE u.user_type = 'supervisor'
            ORDER BY u.full_name
        """)
        
        supervisors = cur.fetchall()
        return pd.DataFrame(supervisors)
    except Exception as e:
        st.error(f"Error fetching supervisors: {e}")
        return pd.DataFrame()
    finally:
        if conn:
            cur.close()
            conn.close()

def get_all_matches():
    """Retrieve all matches from matching_history."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor(cursor_factory=RealDictCursor)

        cur.execute("""
            SELECT 
                mh.id,
                su.full_name AS student_name,
                sp.full_name AS supervisor_name,
                mh.final_score,
                mh.created_at
            FROM matching_history mh
            JOIN users su ON mh.student_id = su.id
            JOIN users sp ON mh.supervisor_id = sp.id
            ORDER BY mh.created_at DESC
        """)
        
        matches = cur.fetchall()
        return pd.DataFrame(matches)
    except Exception as e:
        st.error(f"Error fetching matches: {e}")
        return pd.DataFrame()
    finally:
        if conn:
            cur.close()
            conn.close()

def remove_student_from_supervisor(student_id, supervisor_id):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        cur.execute("""
            DELETE FROM matching_history
            WHERE student_id = %s AND supervisor_id = %s
        """, (student_id, supervisor_id))
        
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Error removing student: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            cur.close()
            conn.close()

def move_student_to_supervisor(student_id, new_supervisor_id, old_supervisor_id):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        cur.execute("""
            UPDATE matching_history
            SET supervisor_id = %s
            WHERE student_id = %s AND supervisor_id = %s
        """, (new_supervisor_id, student_id, old_supervisor_id))
        
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Error moving student: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            cur.close()
            conn.close()

