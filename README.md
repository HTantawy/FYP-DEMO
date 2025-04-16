# 📚 Intelligent Student-Supervisor Matcher Web-Application

This is a web-based allocation system that uses NLP and semantic similarity techniques to match students with the most suitable supervisors based on project proposals, research interests, and technical compatibility. The system integrates BERT, SBERT, TF-IDF, and a custom scoring algorithm, all deployed through a user-friendly Streamlit interface.

## 🚀 Features

- AI-powered matching using BERT + SBERT + TF-IDF
-  Role-based dashboards for students, supervisors, and admins
-  Secure authentication using hashed passwords (bcrypt)
-  Admin dashboard for overseeing matches, reassignments, and exports
-  Messaging and supervision request features
-  Integrated Chatbot for project proposal guidance using OpenAI's API
-  Export functionality (PDF and CSV)

## 🛠️ Tech Stack

- **Frontend & UI**: Streamlit  
- **Backend & Logic**: Python  
- **Database**: PostgreSQL  
- **NLP & Matching**: Transformers (BERT, SBERT), Scikit-learn, Sentence Transformers  
- **Visualizations**: Plotly, Matplotlib, Seaborn  
- **Deployment Option**: Streamlit Cloud / Localhost

## 📦 Installation Instructions

### ✅ Prerequisites

- Python 3.8 or later
- PostgreSQL database setup
- API key from OpenAI (for chatbot functionality)

### 🔧 Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/HTantawy/student-supervisor-matcher.git
   cd student-supervisor-matcher
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set environment variables**
   Create a `.env` file in the root folder and include:
   ```
   DB_HOST=localhost
   DB_PORT=5432
   DB_NAME=your_database
   DB_USER=your_username
   DB_PASSWORD=your_password
   OPENAI_API_KEY=your_openai_api_key
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

## 👤 User Roles

### Student
- Submit project proposals
- View top 3 recommended supervisors
- Send supervision requests

### Supervisor
- View incoming requests
- Accept/reject student proposals
- Update research areas

### Admin
- View overall allocation stats
- Manage student-supervisor matches
- Reassign and transfer students
- Export match data







