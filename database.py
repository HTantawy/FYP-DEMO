# database.py
import psycopg2
import streamlit as st
from auth_app import DB_CONFIG, init_db

# database.py
import psycopg2
import streamlit as st
from auth_app import DB_CONFIG, init_db

def clean_duplicate_data():
    """Remove duplicate supervised projects and publications"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        # Delete duplicate projects
        cur.execute("""
            DELETE FROM supervised_projects
            WHERE id IN (
                SELECT id FROM (
                    SELECT id, 
                           ROW_NUMBER() OVER (PARTITION BY supervisor_id, title, year ORDER BY id) as row_num
                    FROM supervised_projects
                ) as duplicates
                WHERE duplicates.row_num > 1
            )
        """)
        
        projects_deleted = cur.rowcount
        
        # Delete duplicate publications
        cur.execute("""
            DELETE FROM supervisor_publications
            WHERE id IN (
                SELECT id FROM (
                    SELECT id, 
                           ROW_NUMBER() OVER (PARTITION BY supervisor_id, title, year ORDER BY id) as row_num
                    FROM supervisor_publications
                ) as duplicates
                WHERE duplicates.row_num > 1
            )
        """)
        
        pubs_deleted = cur.rowcount
        conn.commit()
        
        print(f"Deleted {projects_deleted} duplicate projects and {pubs_deleted} duplicate publications")
        return projects_deleted + pubs_deleted
        
    except Exception as e:
        print(f"Error cleaning duplicates: {e}")
        if conn:
            conn.rollback()
        return 0
    finally:
        if conn:
            cur.close()
            conn.close()

def verify_database():
    """Verify database connection and table existence"""
    conn = None  # Initialize conn to None before the try block
    cur = None   # Initialize cur to None as well
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        # Check if tables exist
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'users'
            );
        """)
        users_exists = cur.fetchone()[0]
        
        if not users_exists:
            print("Users table not found, initializing database...")
            init_db()
        else:
            print("Database verification successful!")
            
    except Exception as e:
        print(f"Database verification failed: {str(e)}")
        init_db()
        
    finally:
        if cur:  # Check if cur exists before closing it
            cur.close()
        if conn:  # Now this check is safe because conn is always defined
            conn.close()

def create_sample_supervisors():
    """Create sample supervisor accounts for testing purposes"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        # Sample supervisors data with realistic research profiles
        sample_supervisors = [
            {
                "email": "jsmith@bham.ac.uk",
                "password": "password123",
                "full_name": "Prof. James Smith",
                "department": "Computer Science",
                "research_interests": "Deep Learning and Computer Vision research with a focus on object detection and image segmentation in autonomous vehicles. Currently developing novel neural network architectures for real-time pedestrian tracking in adverse weather conditions.",
                "expertise": ["Deep Learning", "Computer Vision", "Machine Learning"],
                "preferred_projects": ["Research-Based", "Industry-focused", "Software Development"],
                "max_capacity": 4,
                "office_hours": "Tuesday and Thursday, 14:00-16:00",
                "contact_preferences": "Email or MS Teams",
                "website_url": "https://cs.bham.ac.uk/~smithj",
                "bio": "Leading researcher in computer vision with 15+ years of experience. Previously worked at Google DeepMind and collaborated with automotive industry partners."
            },
            {
                "email": "lchang@bham.ac.uk",
                "password": "password123",
                "full_name": "Dr. Li Chang",
                "department": "Natural Language Processing",
                "research_interests": "Large language models, transformer architectures, and multilingual NLP systems. My current research focuses on reducing hallucinations in generative AI while maintaining coherence and factual accuracy in specialized domains.",
                "expertise": ["NLP", "Machine Learning", "Text Analysis", "Generative artificial intelligence"],
                "preferred_projects": ["Research-Based", "Software Development"],
                "max_capacity": 3,
                "office_hours": "Monday and Wednesday, 10:00-12:00",
                "contact_preferences": "Email for appointments, quick questions on Slack",
                "website_url": "https://nlp.bham.ac.uk/~changl",
                "bio": "Specializing in multilingual NLP systems with experience at Meta AI Research. Published extensively on transformer architectures and their applications in low-resource languages."
            },
            {
                "email": "kpatel@bham.ac.uk",
                "password": "password123",
                "full_name": "Dr. Kiran Patel",
                "department": "Cybersecurity",
                "research_interests": "Network security, penetration testing, and vulnerability assessment with a special focus on critical infrastructure protection. Currently investigating AI-powered attack vectors and advanced persistent threats in cloud environments.",
                "expertise": ["Cybersecurity", "Network Programming", "Cloud Computing"],
                "preferred_projects": ["Software Development", "Research-Based"],
                "max_capacity": 5,
                "office_hours": "Friday, 9:00-15:00",
                "contact_preferences": "Email with [SEC] in subject line",
                "website_url": "https://cyber.bham.ac.uk/~patelk",
                "bio": "Former cybersecurity consultant with experience protecting financial institutions. Certified ethical hacker and regular contributor to major security conferences."
            },
            {
                "email": "mroberts@bham.ac.uk",
                "password": "password123",
                "full_name": "Prof. Maria Roberts",
                "department": "Data Science",
                "research_interests": "Big data analytics, time series prediction, and anomaly detection in financial datasets. My lab develops scalable algorithms for processing massive datasets with applications in fraud detection and algorithmic trading.",
                "expertise": ["Data Science", "Big Data Processing", "Machine Learning", "Algorithm Design"],
                "preferred_projects": ["Industry-focused", "Research-Based"],
                "max_capacity": 4,
                "office_hours": "Tuesday and Thursday, 11:00-13:00",
                "contact_preferences": "Email or in-person during office hours",
                "website_url": "https://datascience.bham.ac.uk/~robertsm",
                "bio": "Leading researcher in financial data science with industry partnerships in the banking sector. Specializes in time series analysis and high-frequency trading algorithms."
            },
            {
                "email": "adiallo@bham.ac.uk",
                "password": "password123",
                "full_name": "Dr. Aminata Diallo",
                "department": "Healthcare AI",
                "research_interests": "AI applications in medical imaging and clinical decision support. Currently developing deep learning models for early detection of diabetic retinopathy and lung cancer from CT scans with explainable AI components.",
                "expertise": ["AI for Healthcare", "Medical Imaging", "Deep Learning", "Clinical Machine Learning"],
                "preferred_projects": ["Research-Based", "Software Development"],
                "max_capacity": 3,
                "office_hours": "Wednesday, 10:00-16:00",
                "contact_preferences": "Email with prior appointment for meetings",
                "website_url": "https://healthai.bham.ac.uk/~dialloa",
                "bio": "Medical doctor turned AI researcher with a focus on bringing machine learning tools to clinical practice. Works closely with the University Hospital for clinical validation."
            },
            {
                "email": "dnguyen@bham.ac.uk",
                "password": "password123",
                "full_name": "Prof. David Nguyen",
                "department": "Robotics and Automation",
                "research_interests": "Robotic manipulation, reinforcement learning for robotic control, and human-robot interaction. My lab focuses on developing dexterous robotic hands capable of fine manipulation tasks through imitation learning.",
                "expertise": ["Robotics", "Machine Learning", "Human-Computer-Interaction"],
                "preferred_projects": ["Hardware/IoT", "Research-Based", "Software Development"],
                "max_capacity": 5,
                "office_hours": "Monday and Friday, 13:00-15:00",
                "contact_preferences": "Email for appointments, lab visits welcome",
                "website_url": "https://robotics.bham.ac.uk/~nguyend",
                "bio": "Director of the Advanced Robotics Lab with industry experience at Boston Dynamics. Focuses on creating robots with human-like dexterity through machine learning."
            },
            {
                "email": "soliver@bham.ac.uk",
                "password": "password123",
                "full_name": "Dr. Sarah Oliver",
                "department": "Software Engineering",
                "research_interests": "Software architecture for distributed systems, microservices optimization, and DevOps practices. Current research includes automated testing frameworks and continuous deployment methodologies for critical systems.",
                "expertise": ["Software Engineering", "Software Engineering and Distributed Computing", "Cloud Computing"],
                "preferred_projects": ["Software Development", "Industry-focused"],
                "max_capacity": 4,
                "office_hours": "Tuesday and Thursday, 14:00-16:00",
                "contact_preferences": "Microsoft Teams or email",
                "website_url": "https://se.bham.ac.uk/~olivers",
                "bio": "Software architect with extensive industry background at AWS. Specializes in building resilient distributed systems and microservices architectures."
            }
        ]
        
        print("Creating sample supervisors...")
        for supervisor in sample_supervisors:
            # Check if supervisor exists
            cur.execute("SELECT id FROM users WHERE email = %s", (supervisor['email'],))
            if not cur.fetchone():
                # Hash password
                from auth_app import hash_password
                password_hash = hash_password(supervisor['password'])
                
                # Create user
                cur.execute("""
                    INSERT INTO users (email, password_hash, full_name, user_type)
                    VALUES (%s, %s::text, %s, %s)
                    RETURNING id
                """, (supervisor['email'], password_hash, supervisor['full_name'], 'supervisor'))
                
                supervisor_id = cur.fetchone()[0]
                
                # Create supervisor profile
                cur.execute("""
                    INSERT INTO supervisor_profiles 
                    (user_id, research_interests, department, expertise, preferred_projects, max_capacity,
                     office_hours, contact_preferences, website_url, bio)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    supervisor_id,
                    supervisor['research_interests'],
                    supervisor['department'],
                    supervisor['expertise'],
                    supervisor['preferred_projects'],
                    supervisor['max_capacity'],
                    supervisor.get('office_hours', ''),
                    supervisor.get('contact_preferences', ''),
                    supervisor.get('website_url', ''),
                    supervisor.get('bio', '')
                ))
                
                # Generate diverse publications based on expertise
                for i in range(3):  # 3 publications per supervisor
                    year = 2023 - i
                    expertise = supervisor['expertise'][i % len(supervisor['expertise'])]
                    
                    # Create varied publication titles
                    if i == 0:
                        title = f"Advances in {expertise}: A Comprehensive Review"
                        pub_type = "Journal Article"
                        venue = f"Journal of {supervisor['department']}"
                    elif i == 1:
                        title = f"Novel {expertise} Framework for {supervisor['department']} Applications"
                        pub_type = "Conference Paper"
                        venue = f"International Conference on {expertise}"
                    else:
                        title = f"Challenges and Opportunities in {expertise}: Case Studies from {supervisor['department']}"
                        pub_type = "Book Chapter"
                        venue = f"Handbook of {supervisor['department']}"
                    
                    # Generate realistic co-authors
                    coauthors = [supervisor['full_name']]
                    coauthors.extend([f"Collaborator {j+1}" for j in range(2)])
                    
                    cur.execute("""
                        INSERT INTO supervisor_publications
                        (supervisor_id, title, authors, year, publication_type, venue, doi, description)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        supervisor_id,
                        title,
                        coauthors,
                        year,
                        pub_type,
                        venue,
                        f"10.1000/j.{supervisor['department'].lower().replace(' ', '')}.{year}.{i+1}",
                        f"This {pub_type.lower()} presents {['significant advancements', 'novel methods', 'a comprehensive analysis'][i]} in {expertise} with applications to {supervisor['department']}. {['The work demonstrates superior performance on benchmark datasets.', 'Results show a 25% improvement over state-of-the-art approaches.', 'Case studies reveal important insights for future research directions.'][i]}"
                    ))
                # Inside the loop that creates publications:
                for i in range(3):  # 3 publications per supervisor
                    year = 2023 - i
                    expertise = supervisor['expertise'][i % len(supervisor['expertise'])]
                    
                    # Create varied publication titles
                    if i == 0:
                        title = f"Advances in {expertise}: A Comprehensive Review"
                        pub_type = "Journal Article"
                        venue = f"Journal of {supervisor['department']}"
                    elif i == 1:
                        title = f"Novel {expertise} Framework for {supervisor['department']} Applications"
                        pub_type = "Conference Paper"
                        venue = f"International Conference on {expertise}"
                    else:
                        title = f"Challenges and Opportunities in {expertise}: Case Studies from {supervisor['department']}"
                        pub_type = "Book Chapter"
                        venue = f"Handbook of {supervisor['department']}"
                    
                    # Check if publication already exists for this supervisor
                    cur.execute("""
                        SELECT id FROM supervisor_publications
                        WHERE supervisor_id = %s AND title = %s AND year = %s
                    """, (
                        supervisor_id,
                        title,
                        year
                    ))
                    
                    if not cur.fetchone():  # Only insert if not exists
                        # Generate realistic co-authors
                        coauthors = [supervisor['full_name']]
                        coauthors.extend([f"Collaborator {j+1}" for j in range(2)])
                        
                        cur.execute("""
                            INSERT INTO supervisor_publications
                            (supervisor_id, title, authors, year, publication_type, venue, doi, description)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        """, (
                            supervisor_id,
                            title,
                            coauthors,
                            year,
                            pub_type,
                            venue,
                            f"10.1000/j.{supervisor['department'].lower().replace(' ', '')}.{year}.{i+1}",
                            f"This {pub_type.lower()} presents {['significant advancements', 'novel methods', 'a comprehensive analysis'][i]} in {expertise} with applications to {supervisor['department']}. {['The work demonstrates superior performance on benchmark datasets.', 'Results show a 25% improvement over state-of-the-art approaches.', 'Case studies reveal important insights for future research directions.'][i]}"
                        ))    
                
               
                project_types = ["Undergraduate", "Masters", "PhD"]
                project_themes = [
                    "Implementation and Evaluation", 
                    "Design and Optimization", 
                    "Analysis and Comparison"
                ]

                for i in range(3):  # 3 projects per supervisor
                    year = 2023 - i
                    expertise = supervisor['expertise'][i % len(supervisor['expertise'])]
                    preferred_project = supervisor['preferred_projects'][i % len(supervisor['preferred_projects'])]
                    theme = project_themes[i]
                    
                    # Ensure truly unique titles for each project
                    if i == 0:
                        title = f"{year} - {theme} of {expertise} for {supervisor['department']} Applications"
                        description = f"This project developed novel algorithms based on {expertise} to address challenges in {supervisor['department']}. The student implemented a working prototype and evaluated it against industry benchmarks."
                        outcome = "The project resulted in a conference paper and a working software prototype."
                    elif i == 1:
                        title = f"{year} - {theme} of Advanced {expertise} Techniques in {preferred_project} Contexts"
                        description = f"A comprehensive investigation of {expertise} techniques applied to {preferred_project} scenarios. The work included both theoretical analysis and practical implementation with real datasets."
                        outcome = "The student secured a position at a leading tech company based on this work."
                    else:
                        title = f"{year} - {theme} of Emerging {expertise} Approaches in {supervisor['department']}"
                        description = f"This project conducted an extensive evaluation of different {expertise} approaches for solving problems in {supervisor['department']}. The student developed a benchmarking framework that is now used by other researchers."
                        outcome = "This work led to a journal publication and open-source software release."    
                    
                    cur.execute("""
                        INSERT INTO supervised_projects
                        (supervisor_id, title, student_name, year, project_type, description, outcome)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (
                        supervisor_id,
                        title,
                        f"Student {['Alice', 'Bob', 'Charlie', 'Dana', 'Eduardo', 'Fatima'][i % 6]} {['Smith', 'Jones', 'Zhang', 'Patel', 'Garcia', 'Kim'][i % 6]}",
                        year,
                        project_types[i % 3],
                        description,
                        outcome
                    ))
                
                print(f"Created supervisor: {supervisor['full_name']}")
        
        conn.commit()
        print(f"Sample supervisors created successfully! Total: {len(sample_supervisors)}")
        return True
    except Exception as e:
        print(f"Error creating sample supervisors: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            cur.close()
            conn.close()

def create_sample_students():
    """Create sample student accounts with supervisor requests for testing purposes"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        # First ensure we have supervisors to match with
        cur.execute("SELECT COUNT(*) FROM users WHERE user_type = 'supervisor'")
        supervisor_count = cur.fetchone()[0]
        
        if supervisor_count == 0:
            print("No supervisors found. Creating sample supervisors first...")
            create_sample_supervisors()
        
        # Get supervisor IDs and expertise for matching
        cur.execute("""
            SELECT u.id, u.full_name, sp.expertise, sp.research_interests, sp.department
            FROM users u
            JOIN supervisor_profiles sp ON u.id = sp.user_id
            WHERE u.user_type = 'supervisor'
        """)
        supervisors = cur.fetchall()
        
        # Sample students data with varied profiles and interests
        sample_students = [
            {
                "email": "jdoe@bham.ac.uk",
                "password": "password123",
                "full_name": "John Doe",
                "course": "MSc Computer Science",
                "year_of_study": 1,
                "projects": [
                    {
                        "title": "Deep Learning for Facial Recognition in Crowded Environments",
                        "description": "This project aims to develop a robust facial recognition system that works effectively in crowded environments with partial occlusions. The approach will use advanced deep learning techniques including transformers and attention mechanisms to improve accuracy in challenging conditions.",
                        "technical_requirements": ["Python", "TensorFlow", "Deep Learning", "Computer Vision", " Data Analysis"],
                        "methodology": "Quantitative",
                        "project_type": ["Research-Based", "Software Development"]
                    }
                ]
            },
            {
                "email": "asmith2@bham.ac.uk",
                "password": "password123",
                "full_name": "Alice Smith",
                "course": "MSc Artificial Intelligence",
                "year_of_study": 1,
                "projects": [
                    {
                        "title": "Generating Medical Reports from X-ray Images Using Large Language Models",
                        "description": "A project exploring the application of large language models to automatically generate preliminary medical reports from X-ray images. The system will combine computer vision techniques with NLP to create accurate, clinically relevant descriptions of medical images.",
                        "technical_requirements": ["Python", "PyTorch", "NLP", "Computer Vision", "Medical Imaging"],
                        "methodology": "Mixed Methods",
                        "project_type": ["Research-Based", "Software Development"]
                    }
                ]
            },
            {
                "email": "rtaylor@bham.ac.uk",
                "password": "password123",
                "full_name": "Ryan Taylor",
                "course": "MSc Cybersecurity",
                "year_of_study": 1,
                "projects": [
                    {
                        "title": "Detecting Zero-Day Vulnerabilities with Machine Learning",
                        "description": "This project will develop a machine learning approach to identify potential zero-day vulnerabilities in software code. Using both static and dynamic code analysis combined with supervised learning algorithms, the system aims to flag code patterns that may lead to security exploits.",
                        "technical_requirements": ["Python", "Machine Learning", "Network Programming", "Cybersecurity"],
                        "methodology": "Quantitative",
                        "project_type": ["Research-Based", "Software Development"]
                    }
                ]
            },
            {
                "email": "mpatel2@bham.ac.uk",
                "password": "password123",
                "full_name": "Mira Patel",
                "course": "MSc Data Science",
                "year_of_study": 1,
                "projects": [
                    {
                        "title": "Predictive Analytics for Stock Market Trends Using Alternative Data",
                        "description": "This project will investigate how alternative data sources like social media sentiment, satellite imagery, and web scraping can enhance traditional stock market prediction models. It will implement a hybrid approach combining traditional time series analysis with deep learning techniques.",
                        "technical_requirements": ["Python", "Big Data Processing", "Machine Learning", "Data Science"],
                        "methodology": "Quantitative",
                        "project_type": ["Industry-focused", "Research-Based"]
                    }
                ]
            }
        ]
        
        print("Creating sample students...")
        created_students = []
        
        for student in sample_students:
            # Check if student exists
            cur.execute("SELECT id FROM users WHERE email = %s", (student['email'],))
            if not cur.fetchone():
                # Hash password
                from auth_app import hash_password
                password_hash = hash_password(student['password'])
                
                # Create user
                cur.execute("""
                    INSERT INTO users (email, password_hash, full_name, user_type)
                    VALUES (%s, %s::text, %s, %s)
                    RETURNING id
                """, (student['email'], password_hash, student['full_name'], 'student'))
                
                student_id = cur.fetchone()[0]
                
                # Create student profile
                cur.execute("""
                    INSERT INTO student_profiles 
                    (user_id, course, year_of_study)
                    VALUES (%s, %s, %s)
                """, (
                    student_id,
                    student['course'],
                    student['year_of_study']
                ))
                
                created_students.append({
                    'id': student_id, 
                    'name': student['full_name'],
                    'projects': student['projects']
                })
                
                print(f"Created student: {student['full_name']}")
        
        # Create supervisor requests - match each student with 2-3 appropriate supervisors
        from student_supervisor import AdvancedSupervisorMatcher
        matcher = AdvancedSupervisorMatcher()
        
        print("Creating supervisor requests and match history...")
        for student in created_students:
            for project in student['projects']:
                # Prepare data for matching
                student_data = {
                    'student_name': student['name'],
                    'project_title': project['title'],
                    'project_description': (
                        f"{project['description']}\n"
                        f"Technical requirements: {', '.join(project['technical_requirements'])}.\n"
                        f"Research methodology: {project['methodology']}."
                        f"Project type: {', '.join(project['project_type'])}."
                    ),
                    'project_type': project['project_type']
                }
                
                # Create supervisors data format expected by matcher
                supervisors_data = [
                    {
                        'id': supervisor[0],
                        'name': supervisor[1], 
                        'interests': supervisor[3],
                        'expertise': supervisor[2],
                        'department': supervisor[4],
                        'project_types': ['Research-Based', 'Software Development']  # Default for simplicity
                    }
                    for supervisor in supervisors
                ]
                
                # Get matches
                matches = matcher.match_supervisors(student_data, supervisors_data)
                
                # Send requests to top 3 matching supervisors (if available)
                for i, match in enumerate(matches[:min(3, len(matches))]):
                    supervisor_id = next(
                        (s['id'] for s in supervisors_data if s['name'] == match['supervisor_name']), 
                        None
                    )
                    
                    if supervisor_id:
                        # Create a supervisor request with varied statuses
                        status = ['pending', 'accepted', 'rejected'][min(i, 2)]  # First match accepted, second pending, third rejected
                        
                        cur.execute("""
                            INSERT INTO supervisor_requests 
                            (student_id, supervisor_id, project_title, project_description, matching_score, status)
                            VALUES (%s, %s, %s, %s, %s, %s)
                            ON CONFLICT (student_id, supervisor_id) DO NOTHING
                        """, (
                            student['id'],
                            supervisor_id,
                            project['title'],
                            student_data['project_description'],
                            match['final_score'],
                            status
                        ))
                        
                        # Create matching history
                        cur.execute("""
                            INSERT INTO matching_history 
                            (student_id, supervisor_id, final_score, research_alignment, 
                             methodology_match, technical_skills, domain_knowledge, project_type_match)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        """, (
                            student['id'],
                            supervisor_id,
                            match['final_score'],
                            match['detailed_scores']['research_alignment'],
                            match['detailed_scores']['methodology_match'],
                            match['detailed_scores']['technical_skills'],
                            match['detailed_scores']['domain_knowledge'],
                            match['detailed_scores'].get('project_type_match', 0.0)
                        ))
                
                # For the first match, create some messages too
                if matches and len(matches) > 0:
                    first_supervisor_id = next(
                        (s['id'] for s in supervisors_data if s['name'] == matches[0]['supervisor_name']), 
                        None
                    )
                    
                    if first_supervisor_id:
                        # Create a conversation between student and supervisor
                        messages = [
                            (student['id'], first_supervisor_id, f"Hello Professor, I'm interested in working with you on my project '{project['title']}'. Would you be available to discuss it further?"),
                            (first_supervisor_id, student['id'], f"Hello {student['name']}, thank you for your interest. Your project sounds interesting. Could you tell me more about your background in {project['technical_requirements'][0]}?"),
                            (student['id'], first_supervisor_id, f"I've completed coursework in {project['technical_requirements'][0]} and have worked on several personal projects. I'm particularly interested in your research on {matches[0]['detailed_scores']}.")
                        ]
                        
                        for sender_id, receiver_id, message_text in messages:
                            cur.execute("""
                                INSERT INTO messages
                                (sender_id, receiver_id, message_text, is_read)
                                VALUES (%s, %s, %s, %s)
                            """, (
                                sender_id, 
                                receiver_id, 
                                message_text,
                                sender_id != student['id']  # Student messages unread, supervisor messages read
                            ))
        
        conn.commit()
        print("Sample students with matching requests created successfully!")
        return True
    except Exception as e:
        print(f"Error creating sample students: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            cur.close()
            conn.close()

def verify_database():
    """Verify database connection and table existence with optimized sample data loading"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        # Check if tables exist
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'users'
            );
        """)
        users_exists = cur.fetchone()[0]
        
        if not users_exists:
            # First time setup - tables don't exist
            print("Users table not found, initializing database...")
            init_db()  # Create all tables
            
            # Create sample data
            create_sample_supervisors()
            create_sample_students()
        else:
            print("Database verification successful!")
            
            # Check if we have any supervisors (better indicator for sample data)
            cur.execute("SELECT COUNT(*) FROM users WHERE user_type = 'supervisor'")
            supervisor_count = cur.fetchone()[0]
            
            if supervisor_count == 0:
                # No supervisors found, need to create sample data
                print("No supervisors found. Creating sample data...")
                create_sample_supervisors()
                create_sample_students()
            else:
                print(f"Found {supervisor_count} supervisors - sample data already exists")
                
    except Exception as e:
        # Handle database connection errors
        print(f"Database verification failed: {str(e)}")
        init_db()
        create_sample_supervisors()
        create_sample_students()
        
    finally:
        if conn:
            cur.close()
            conn.close()

def reset_database():
    """Reset the database for testing purposes"""
    try:
        # Reinitialize the database
        init_db()
        
        # Create sample data
        create_sample_supervisors()
        create_sample_students()
        
        return True
    except Exception as e:
        print(f"Error resetting database: {e}")
        return False

def add_admin_reset_controls(st_obj):
    """Add reset controls to the admin dashboard"""
    with st_obj.expander("Advanced Admin Controls", expanded=False):
        st_obj.warning("⚠️ These operations can permanently modify the database. Use with caution.")
        
        col1, col2 = st_obj.columns(2)
        
        with col1:
            if st_obj.button("Reset All Sample Data", key="reset_all", use_container_width=True):
                if reset_database():
                    st_obj.success("Database reset successfully with fresh sample data!")
                    st_obj.info("Please log out and log back in to see the changes.")
                else:
                    st_obj.error("Failed to reset database.")
        
        with col2:
            if st_obj.button("Reset Only Student Data", key="reset_students", use_container_width=True):
                try:
                    # Delete only student-related data
                    conn = psycopg2.connect(**DB_CONFIG)
                    cur = conn.cursor()
                    
                    # Get all student IDs
                    cur.execute("SELECT id FROM users WHERE user_type = 'student'")
                    student_ids = [row[0] for row in cur.fetchall()]
                    
                    if student_ids:
                        # Delete related data from various tables
                        for student_id in student_ids:
                            cur.execute("DELETE FROM matching_history WHERE student_id = %s", (student_id,))
                            cur.execute("DELETE FROM supervisor_requests WHERE student_id = %s", (student_id,))
                            cur.execute("DELETE FROM messages WHERE sender_id = %s OR receiver_id = %s", 
                                       (student_id, student_id))
                            cur.execute("DELETE FROM notifications WHERE user_id = %s", (student_id,))
                        
                        # Delete student profiles and users
                        cur.execute("DELETE FROM student_profiles WHERE user_id IN (SELECT id FROM users WHERE user_type = 'student')")
                        cur.execute("DELETE FROM users WHERE user_type = 'student'")
                        
                        conn.commit()
                        
                        # Create new sample students
                        create_sample_students()
                        
                        st_obj.success("Student data reset successfully!")
                        st_obj.info("Please refresh to see the changes.")
                    else:
                        st_obj.info("No student data to reset.")
                        
                except Exception as e:
                    st_obj.error(f"Error resetting student data: {e}")
                    if conn:
                        conn.rollback()
                finally:
                    if conn:
                        cur.close()
                        conn.close()
                        
        st_obj.markdown("---")
        
        col1, col2 = st_obj.columns(2)
        with col1:
            st_obj.write("**Current Database Stats:**")
            
            # Count users by type
            conn = psycopg2.connect(**DB_CONFIG)
            cur = conn.cursor()
            
            cur.execute("SELECT user_type, COUNT(*) FROM users GROUP BY user_type")
            user_counts = cur.fetchall()
            
            for user_type, count in user_counts:
                st_obj.write(f"• {user_type.title()}s: {count}")
            
            # Count supervisor requests
            cur.execute("SELECT status, COUNT(*) FROM supervisor_requests GROUP BY status")
            request_counts = cur.fetchall()
            
            st_obj.write("**Supervisor Requests:**")
            for status, count in request_counts:
                st_obj.write(f"• {status.title()}: {count}")
                
            cur.close()
            conn.close()