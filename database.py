
import psycopg2
import streamlit as st
from auth_app import DB_CONFIG, init_db


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
        if cur:  
            cur.close()
        if conn:  
            conn.close()

def create_sample_supervisors():
    """Create sample supervisor accounts for testing purposes"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        # Sample supervisors data with realistic research profiles
        sample_supervisors = [
            {
                "email": "Martin_12@bham.ac.uk",
                "password": "password123",
                "full_name": "Prof. Martin Escardo",
                "department": "Computer Science",
                "research_interests": "Exploring how functional programming paradigms (e.g., Haskell, OCaml, F#) can be leveraged to implement machine learning models efficiently.",
                "expertise": ["Functional Programming", "Type Theory",],
                "preferred_projects": ["Research-Based", "Industry-focused", "Software Development"],
                "max_capacity": 9,
                "office_hours": "Tuesday and Thursday, 14:00-16:00",
                "contact_preferences": "Email or MS Teams",
                "website_url": "https://cs.bham.ac.uk/~smithj",
                "bio": "Leading researcher in computer vision with 15+ years of experience. Previously worked at Google DeepMind and collaborated with automotive industry partners."
            },
            {
                "email": "ruchit_31@bham.ac.uk",
                "password": "password123",
                "full_name": "Dr. Ruchit Agrawal",
                "department": "Natural Language Processing",
                "research_interests": "Natural language processing; Multi-modal deep learning; Generative artificial intelligence using transformer-based models (e.g., ChatGPT);Music informatics; Healthcare artificial intelligence; Clinical Machine learning; Audio signal processing using machine learning",
                "expertise": ["NLP", "Machine Learning", "Text Analysis", "Generative artificial intelligence"],
                "preferred_projects": ["Research-Based", "Software Development"],
                "max_capacity": 10,
                "office_hours": "Monday and Wednesday, 10:00-12:00",
                "contact_preferences": "Email for appointments, quick questions on Slack",
                "website_url": "https://nlp.bham.ac.uk/~changl",
                "bio": "Specializing in multilingual NLP systems with experience at Meta AI Research. Published extensively on transformer architectures and their applications in low-resource languages.",
            },
            {
                "email": "Mark16@bham.ac.uk",
                "password": "password123",
                "full_name": "Dr. Mark Lee",
                "department": "Artificial Intelligence",
                "research_interests": "I am happy to supervise any AI/natural language processing project. In particular, I am currently very interested in the use of corpora (large bodies of text) and empirical methods to develop robust techniques and systems. The following are a few concrete projects i have in mind but I am open to other ideas.Affect & emotion detection in TextCertain keywords can be associated with different emotions. For example words such as good, kind and happy  are useful indicators that a newspaper story has some positive connotation. I'd like to supervise a project which uses statistical methods to rate newspaper stories (or movie reviews). One interesting application would be a measure of subjectivity versus objectivity in newspaper reporting. 2.Text Summarisation Techniques More and more newspapers now publish their pages on the web. However, often users are too busy to spend the time reading a newspaper. What is required is a summarisation agent which fetches news from the internet and then summarizes it for the user.",
                "expertise": ["NLP", "Sentiment Analysis", "Text Analysis"],
                "preferred_projects": ["Software Development", "Research-Based"],
                "max_capacity": 8,
                "office_hours": "Friday, 9:00-15:00",
                "contact_preferences": "Email with [SEC] in subject line",
                "website_url": "https://markglee.github.io/student-projects.html",
                "bio": "Former cybersecurity consultant with experience protecting financial institutions. Certified ethical hacker and regular contributor to major security conferences."
            },
            {
                "email": "p.lehre@bham.ac.uk",
                "password": "password123",
                "full_name": "Prof. Per Kristian Lehre",
                "department": "Theory of Evolutionary Computation",
                "research_interests": "My research interests are in theory of evolutionary computation, population genetics, randomised algorithms and related topics.",
                "expertise": ["Evolutionary Algorithms", "Neural Networks","Algorithm Design"],
                "preferred_projects": ["Industry-focused", "Research-Based"],
                "max_capacity": 5,
                "office_hours": "Tuesday and Thursday, 11:00-13:00",
                "contact_preferences": "Email or in-person during office hours",
                "website_url": "https://datascience.bham.ac.uk/~robertsm",
                "bio": "Leading researcher in financial data science with industry partnerships in the banking sector. Specializes in time series analysis and high-frequency trading algorithms."
            },
            {
                "email": "h.mukhtar@bham.ac.uk",
                "password": "password123",
                "full_name": "Dr. Hamid Mukhtar",
                "department": "Machine Learning",
                "research_interests": "Using data analytics to identify the word origin for Urdu language words. Urdu language is considered a derivate of Arabic, Persian, and some other languages. Through data analytics techniques, we can identify which word in Urdu comes from which other language or is derived from it.Pre-requisites: The ability to read and understand Urdu and the ability to read Arabic and Persian. Good knowledge of Python and Pandas. Knowledge of NLP is a plus.",
                "expertise": ["Data Science", "NLP", "Deep Learning"],
                "preferred_projects": ["Research-Based", "Software Development"],
                "max_capacity": 9,
                "office_hours": "Wednesday, 10:00-16:00",
                "contact_preferences": "Email with prior appointment for meetings",
                "website_url": "https://yahamid0.wordpress.com/2025/01/20/project-ideas-2025/",
                "bio": "Medical doctor turned AI researcher with a focus on bringing machine learning tools to clinical practice. Works closely with the University Hospital for clinical validation."
            },
            {
                "email": "rbeale@bham.ac.uk",
                "password": "password123",
                "full_name": "Prof. Russell Beale",
                "department": "Human-Computer Interaction",
                "research_interests": "My research focuses on the intersection of Human-Computer Interaction (HCI), Artificial Intelligence, and Ubiquitous Computing, exploring how intelligent systems can enhance user experience, behavior change, and social interactions. I am particularly interested in AI-driven interactive systems, adaptive user interfaces, and ambient intelligence, designing technologies that seamlessly integrate into everyday life. My work also involves social media analysis, leveraging machine learning and natural language processing to understand online behaviors, sentiment trends, and digital communication patterns. Additionally, I investigate behavioral psychology in technology design, studying how intelligent systems can encourage positive user behaviors, particularly in health, productivity, and social well-being. Through my research, I aim to bridge the gap between cognitive science, design, and AI to create novel, user-centered technologies that drive meaningful engagement and innovation.",
                "expertise": ["Social Media Analysis", "Mobile and ubiquitous systems", "Human-Computer-Interaction"],
                "preferred_projects": ["Research-Based", "Software Development"],
                "max_capacity": 8,
                "office_hours": "Monday and Friday, 13:00-15:00",
                "contact_preferences": "Email for appointments, lab visits welcome",
                "website_url": "https://robotics.bham.ac.uk/~nguyend",
                "bio": "Director of the HCI module "
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
            },
            {
                "email": "r.bahsoon@bham.ac.uk",
                "password": "password123",
                "full_name": "Dr. Rami Bahsoon",
                "department": " Autonomous Software Engineering",
                "research_interests": "My research interests includes Software engineering; Technical debt management; Security software engineering; Internet of things; Cloud computing; Digital twins software architectures; Compliance management in smart systems; Software engineering ethics; Software economics; Software architecture evaluation methods  Software architecture for distributed systems, microservices optimization, and DevOps practices. Current research includes automated testing frameworks and continuous deployment methodologies for critical systems.",
                "expertise": ["Software Engineering", "Software Engineering and Distributed Computing"],
                "preferred_projects": ["Software Development", "Industry-focused"],
                "max_capacity": 10,
                "office_hours": "Tuesday and Thursday, 9:00-16:00",
                "contact_preferences": "Microsoft Teams or email",
                "website_url": "https://temp-mpnnxwddlklcummxbsap.webadorsite.com/",
                "bio": "I am Reader in Distributed and Autonomous Software Engineering at the School of Computer Science, the University of Birmingham, UK and a founding member of its Sociotechnical Systems Research Group"
            }

        ]
        
        print("Creating sample supervisors...")
        for supervisor in sample_supervisors:
            
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
                
                
                for i in range(3):  
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
                
                for i in range(3):  
                    year = 2023 - i
                    expertise = supervisor['expertise'][i % len(supervisor['expertise'])]
                    
                    
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
                    
                    
                    cur.execute("""
                        SELECT id FROM supervisor_publications
                        WHERE supervisor_id = %s AND title = %s AND year = %s
                    """, (
                        supervisor_id,
                        title,
                        year
                    ))
                    
                    if not cur.fetchone():  
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
        
        
        cur.execute("SELECT COUNT(*) FROM users WHERE user_type = 'supervisor'")
        supervisor_count = cur.fetchone()[0]
        
        if supervisor_count == 0:
            print("No supervisors found. Creating sample supervisors first...")
            create_sample_supervisors()
        
        
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
                        "project_type": ["Research-Based", "Software Development"]
                    }
                ]
            },
            {
                "email": "mohamedelsayed2@bham.ac.uk",
                "password": "password123",
                "full_name": "Mohamed Moustafa",
                "course": "MSc Artificial Intelligence",
                "year_of_study": 2,
                "projects": [
                    {
                        "title": "Generating Medical Reports from X-ray Images Using Large Language Models",
                        "description": "A project exploring the application of large language models to automatically generate preliminary medical reports from X-ray images. The system will combine computer vision techniques with NLP to create accurate, clinically relevant descriptions of medical images.",
                        "technical_requirements": ["Python", "PyTorch", "NLP", "Computer Vision", "Medical Imaging"],
                        "project_type": ["Research-Based", "Software Development"]
                    }
                ]
            },
                {
                "email": "nour12@bham.ac.uk",
                "password": "password123",
                "full_name": "Nour Tantawy",
                "course": "MSc Computer Science",
                "year_of_study": 4,
                "projects": [
                    {
                        "title": "Document Summarization Using NLP and Machine Learning",
                        "description": "This project focuses on building an NLP pipeline for automatic document summarization using sequence-to-sequence models (e.g., T5 or Pegasus). The methodology involves preprocessing long text datasets, training abstractive summarization models, and evaluating results with metrics like ROUGE and METEOR.Expected outcomes include an NLP pipeline, summarized text outputs, and a web app for interactive summarization.",
                        "technical_requirements": ["Python", "PyTorch", "NLP"],
                        "project_type": ["Industry-focused", "Software Development"]
                    }
                ]
            },
            {
                "email": "ahmed_12@bham.ac.uk",
                "password": "password123",
                "full_name": "Ahmed Badawy",
                "course": "MSc Data Science",
                "year_of_study": 1,
                "projects": [
                    {
                        "title": "Predictive Analytics for Stock Market Trends Using Alternative Data",
                        "description": "This project will investigate how alternative data sources like social media sentiment, satellite imagery, and web scraping can enhance traditional stock market prediction models. It will implement a hybrid approach combining traditional time series analysis with deep learning techniques.",
                        "technical_requirements": ["Python", "Big Data Processing", "Machine Learning", "Data Science"],
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
                
                
                cur.execute("""
                    INSERT INTO users (email, password_hash, full_name, user_type)
                    VALUES (%s, %s::text, %s, %s)
                    RETURNING id
                """, (student['email'], password_hash, student['full_name'], 'student'))
                
                student_id = cur.fetchone()[0]
                
                
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
                            (student_id, supervisor_id, final_score, research_alignment, technical_skills, domain_knowledge, project_type_match)
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """, (
                            student['id'],
                            supervisor_id,
                            match['final_score'],
                            match['detailed_scores']['research_alignment'],
                            match['detailed_scores']['technical_skills'],
                            match['detailed_scores']['domain_knowledge'],
                            match['detailed_scores'].get('project_type_match', 0.0)
                        ))
                
                if matches and len(matches) > 0:
                    first_supervisor_id = next(
                        (s['id'] for s in supervisors_data if s['name'] == matches[0]['supervisor_name']), 
                        None
                    )
                    
                    if first_supervisor_id:
                        
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
           
            print("Users table not found, initializing database...")
            init_db()  
            
            
            create_sample_supervisors()
            create_sample_students()
        else:
            print("Database verification successful!")
            
            
            cur.execute("SELECT COUNT(*) FROM users WHERE user_type = 'supervisor'")
            supervisor_count = cur.fetchone()[0]
            
            if supervisor_count == 0:
                
                print("No supervisors found. Creating sample data...")
                create_sample_supervisors()
                create_sample_students()
            else:
                print(f"Found {supervisor_count} supervisors - sample data already exists")
                
    except Exception as e:
        
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
        
        init_db()
        
        
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
                    
                    conn = psycopg2.connect(**DB_CONFIG)
                    cur = conn.cursor()
                    
                    
                    cur.execute("SELECT id FROM users WHERE user_type = 'student'")
                    student_ids = [row[0] for row in cur.fetchall()]
                    
                    if student_ids:
                        
                        for student_id in student_ids:
                            cur.execute("DELETE FROM matching_history WHERE student_id = %s", (student_id,))
                            cur.execute("DELETE FROM supervisor_requests WHERE student_id = %s", (student_id,))
                            cur.execute("DELETE FROM messages WHERE sender_id = %s OR receiver_id = %s", 
                                       (student_id, student_id))
                            cur.execute("DELETE FROM notifications WHERE user_id = %s", (student_id,))
                        
                        
                        cur.execute("DELETE FROM student_profiles WHERE user_id IN (SELECT id FROM users WHERE user_type = 'student')")
                        cur.execute("DELETE FROM users WHERE user_type = 'student'")
                        
                        conn.commit()
                        
                        
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
            
            
            conn = psycopg2.connect(**DB_CONFIG)
            cur = conn.cursor()
            
            cur.execute("SELECT user_type, COUNT(*) FROM users GROUP BY user_type")
            user_counts = cur.fetchall()
            
            for user_type, count in user_counts:
                st_obj.write(f"• {user_type.title()}s: {count}")
            
            
            cur.execute("SELECT status, COUNT(*) FROM supervisor_requests GROUP BY status")
            request_counts = cur.fetchall()
            
            st_obj.write("**Supervisor Requests:**")
            for status, count in request_counts:
                st_obj.write(f"• {status.title()}: {count}")
                
            cur.close()
            conn.close()