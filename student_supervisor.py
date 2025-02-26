# Deep Learning and BERT
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

# Data processing and scientific computing
import numpy as np
import pandas as pd
from collections import defaultdict

# Machine Learning and NLP
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler

# NLTK components
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.punkt import PunktSentenceTokenizer

# Data structures and typing
from typing import List, Dict, Tuple, Any, Optional, Union
import json

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# System utilities
import os
import logging
from datetime import datetime

# Refined domain weights based on importance
DOMAIN_WEIGHTS = {
    'research_alignment': 0.30,  # Reduced slightly to balance with other factors
    'methodology_match': 0.25,
    'technical_skills': 0.20,  # Increased due to importance for project success
    'domain_knowledge': 0.15,  # Increased for better subject matter matching
    'project_type_match': 0.10  # Slightly reduced but still significant
}

def ensure_nltk_resources():
    """Ensure all required NLTK resources are downloaded"""
    import nltk
    import ssl
    import os

    # Handle SSL certificate verification issues
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    # Create default directory if it doesn't exist
    nltk_data_dir = os.path.expanduser('~/nltk_data')
    if not os.path.exists(nltk_data_dir):
        os.makedirs(nltk_data_dir)

    resources = [
        'punkt', 'punkt_tab', 'stopwords', 'wordnet',
        'averaged_perceptron_tagger', 'omw-1.4'
    ]

    for resource in resources:
        try:
            print(f"Checking/Downloading {resource}...")
            nltk.download(resource, quiet=True, download_dir=nltk_data_dir)
        except Exception as e:
            print(f"First attempt to download {resource} failed: {str(e)}")
            try:
                nltk.download(resource, download_dir=nltk_data_dir)
            except Exception as e2:
                print(f"Second attempt to download {resource} failed: {str(e2)}")

def check_nltk_data():
    """Verify NLTK data is properly installed"""
    import nltk
    for resource in ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']:
        try:
            nltk.data.find(f'tokenizers/{resource}')
            print(f"{resource} is properly installed")
        except LookupError:
            print(f"Warning: {resource} is not installed properly")
            return False
    return True

class AdvancedSupervisorMatcher:
    def __init__(self):
        # Initialize NLTK resources
        ensure_nltk_resources()
        if not check_nltk_data():
            print("Warning: Some NLTK resources might not be properly installed")
        
        # Initialize NLTK components
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Initialize both BERT and SBERT models
        print("Loading BERT and SBERT models...")
        # Original BERT
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = AutoModel.from_pretrained('bert-base-uncased')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Initialize SBERT
        self.sbert_model = SentenceTransformer('all-mpnet-base-v2')
        self.sbert_model = self.sbert_model.to(self.device)
        
        # Enhanced TF-IDF vectorizer
        self.tfidf = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=1.0,
            max_features=None
        )
        
        # Domain weights
        self.domain_weights = DOMAIN_WEIGHTS
        
        # Technical skills dictionary
        self.technical_skills = {
            'programming': {
                'languages': ['python', 'java', 'c++', 'javascript', 'r', 'matlab', 'scala', 'ruby', 'php'],
                'web': ['html', 'css', 'react', 'angular', 'vue', 'django', 'flask', 'node.js'],
                'mobile': ['android', 'ios', 'flutter', 'react native', 'swift', 'kotlin']
            },
            'machine_learning': {
                'frameworks': ['tensorflow', 'pytorch', 'keras', 'scikit-learn', 'xgboost', 'lightgbm', 'Medical Imaging'],
                'concepts': ['neural networks', 'deep learning', 'reinforcement learning', 'computer vision', 
                           'nlp', 'transformers', 'gan', 'clustering', 'classification']
            },
            'data_analysis': {
                'tools': ['pandas', 'numpy', 'scipy', 'matplotlib', 'seaborn', 'tableau', 'power bi'],
                'techniques': ['statistical analysis', 'data mining', 'time series', 'regression', 
                             'hypothesis testing', 'a/b testing', 'data visualization']
            },
            'cloud': {
                'platforms': ['aws', 'azure', 'google cloud', 'docker', 'kubernetes'],
                'concepts': ['cloud computing', 'microservices', 'serverless', 'devops', 'ci/cd']
            }
        }
        
        # Methodology terms
        self.methodology_terms = {
            'quantitative': {
                'statistical': ['regression analysis', 'statistical modeling', 'hypothesis testing', 
                              'quantitative methods', 'statistical analysis'],
                'experimental': ['controlled experiment', 'randomized trial', 'quasi-experimental'],
                'computational': ['simulation', 'numerical analysis', 'mathematical modeling']
            },
            'qualitative': {
                'methods': ['case study', 'ethnography', 'grounded theory', 'phenomenology'],
                'data_collection': ['interviews', 'focus groups', 'observation', 'document analysis'],
                'analysis': ['thematic analysis', 'content analysis', 'discourse analysis']
            },
            'mixed_methods': {
                'approaches': ['mixed methods', 'triangulation', 'sequential explanatory', 
                             'concurrent nested', 'transformative'],
                'integration': ['data integration', 'methodological integration', 'theoretical integration']
            }
        }
        
        # Initialize scalers
        self.scalers = {
            'research': MinMaxScaler(),
            'methodology': MinMaxScaler(),
            'technical': MinMaxScaler(),
            'domain': MinMaxScaler(),
            'project': MinMaxScaler()
        }

    def preprocess_text(self, text: str) -> str:
        """Enhanced text preprocessing with lemmatization and special handling"""
        # Tokenize and convert to lower case
        tokens = word_tokenize(text.lower())
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and token.isalnum()]
        
        # Handle common abbreviations
        abbreviations = {
            'ml': 'machine learning',
            'ai': 'artificial intelligence',
            'dl': 'deep learning',
            'nlp': 'natural language processing',
            'cv': 'computer vision'
        }
        
        processed_tokens = []
        for token in tokens:
            if token in abbreviations:
                processed_tokens.extend(abbreviations[token].split())
            else:
                processed_tokens.append(token)
        
        return ' '.join(processed_tokens)

    def get_bert_embedding(self, text: str) -> np.ndarray:
        """Get BERT embeddings with attention masking and pooling"""
        text = self.preprocess_text(text)
        
        inputs = self.tokenizer(text, padding=True, truncation=True,
                              max_length=512, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state
            
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            
            pooled_embeddings = sum_embeddings / sum_mask
            
        return pooled_embeddings.cpu().numpy()

    def get_sbert_embedding(self, text: str) -> np.ndarray:
        """Get SBERT embeddings for text"""
        text = self.preprocess_text(text)
        with torch.no_grad():
            embedding = self.sbert_model.encode(text, convert_to_numpy=True)
        return embedding

    def calculate_research_alignment(self, student_desc: str, supervisor_interests: str) -> float:
        """Calculate semantic similarity using both BERT and SBERT"""
        # Get BERT embeddings and similarity
        bert_student_emb = self.get_bert_embedding(student_desc)
        bert_supervisor_emb = self.get_bert_embedding(supervisor_interests)
        bert_similarity = float(cosine_similarity(bert_student_emb, bert_supervisor_emb)[0][0])
        
        # Get SBERT embeddings and similarity
        sbert_student_emb = self.get_sbert_embedding(student_desc)
        sbert_supervisor_emb = self.get_sbert_embedding(supervisor_interests)
        sbert_similarity = float(cosine_similarity(
            sbert_student_emb.reshape(1, -1),
            sbert_supervisor_emb.reshape(1, -1)
        )[0][0])
        
        # Combined similarity score
        combined_similarity = 0.4 * bert_similarity + 0.6 * sbert_similarity
        
        # TF-IDF term overlap
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform([student_desc, supervisor_interests])
        terms1 = set(vectorizer.get_feature_names_out()[tfidf_matrix[0].nonzero()[1]])
        terms2 = set(vectorizer.get_feature_names_out()[tfidf_matrix[1].nonzero()[1]])
        term_overlap = len(terms1.intersection(terms2)) / max(len(terms1.union(terms2)), 1)
        
        return 0.8 * combined_similarity + 0.2 * term_overlap

    def extract_technical_skills(self, text: str) -> Dict[str, List[str]]:
        """Extract and categorize technical skills"""
        text_lower = self.preprocess_text(text)
        found_skills = defaultdict(lambda: defaultdict(list))
        
        for category, subcategories in self.technical_skills.items():
            for subcategory, skills in subcategories.items():
                for skill in skills:
                    if skill in text_lower or any(term in text_lower for term in skill.split()):
                        found_skills[category][subcategory].append(skill)
        
        return dict(found_skills)

    def calculate_methodology_match(self, student_desc: str, supervisor_interests: str) -> float:
        """Calculate methodology alignment"""
        student_methods = self._extract_methodology(student_desc)
        supervisor_methods = self._extract_methodology(supervisor_interests)
        
        if not student_methods or not supervisor_methods:
            return 0.5
        
        total_score = 0
        weights = {'quantitative': 0.4, 'qualitative': 0.3, 'mixed_methods': 0.3}
        
        for category, weight in weights.items():
            if category in student_methods and category in supervisor_methods:
                student_subs = set(student_methods[category])
                supervisor_subs = set(supervisor_methods[category])
                overlap = len(student_subs & supervisor_subs)
                total = len(student_subs | supervisor_subs)
                
                if total > 0:
                    total_score += weight * (overlap / total)
        
        return total_score

    def _extract_methodology(self, text: str) -> Dict[str, set]:
        """Extract methodology terms"""
        text_lower = self.preprocess_text(text)
        found_methods = defaultdict(set)
        
        for category, subcategories in self.methodology_terms.items():
            for subcategory, terms in subcategories.items():
                for term in terms:
                    if term in text_lower:
                        found_methods[category].add(subcategory)
        
        return dict(found_methods)

    def calculate_domain_knowledge(self, student_desc: str, supervisor_interests: str) -> float:
        """Calculate domain knowledge using both BERT and SBERT"""
        try:
            student_text = self.preprocess_text(student_desc)
            supervisor_text = self.preprocess_text(supervisor_interests)
            
            tfidf_matrix = self.tfidf.fit_transform([student_text, supervisor_text])
            feature_names = self.tfidf.get_feature_names_out()
            
            student_terms = set(feature_names[i] for i in tfidf_matrix[0].nonzero()[1])
            supervisor_terms = set(feature_names[i] for i in tfidf_matrix[1].nonzero()[1])
            
            if not student_terms or not supervisor_terms:
                return 0.5
            
            # Get BERT similarity
            bert_sim = float(cosine_similarity(
                self.get_bert_embedding(student_desc),
                self.get_bert_embedding(supervisor_interests)
            )[0][0])
            
            # Get SBERT similarity
            sbert_sim = float(cosine_similarity(
                self.get_sbert_embedding(student_desc).reshape(1, -1),
                self.get_sbert_embedding(supervisor_interests).reshape(1, -1)
            )[0][0])
            
            # Combined semantic similarity
            semantic_sim = 0.4 * bert_sim + 0.6 * sbert_sim
            
            # Combine with term overlap
            term_overlap = len(student_terms & supervisor_terms) / len(student_terms | supervisor_terms)
            return 0.7 * semantic_sim + 0.3 * term_overlap
            
        except ValueError as e:
            print(f"TF-IDF error: {e}")
            # Fallback to combined BERT and SBERT similarity
            bert_sim = float(cosine_similarity(
                self.get_bert_embedding(student_desc),
                self.get_bert_embedding(supervisor_interests)
            )[0][0])
            sbert_sim = float(cosine_similarity(
                self.get_sbert_embedding(student_desc).reshape(1, -1),
                self.get_sbert_embedding(supervisor_interests).reshape(1, -1)
            )[0][0])
            return 0.4 * bert_sim + 0.6 * sbert_sim

    def calculate_project_type_match(self, student_desc: str, supervisor_project_types: List[str]) -> float:
        """Calculate project type match using both BERT and SBERT"""
        if not supervisor_project_types:
            return 0.5
        
        student_desc_lower = self.preprocess_text(student_desc)
        
        # Calculate direct matches
        direct_matches = 0
        for proj_type in supervisor_project_types:
            proj_type_lower = proj_type.lower().replace('-', ' ')
            if proj_type_lower in student_desc_lower:
                direct_matches += 1
        direct_score = direct_matches / len(supervisor_project_types)
        
        # Calculate semantic similarity using both models
        bert_scores = []
        sbert_scores = []
        for proj_type in supervisor_project_types:
            # BERT similarity
            bert_sim = float(cosine_similarity(
                self.get_bert_embedding(student_desc),
                self.get_bert_embedding(proj_type)
            )[0][0])
            bert_scores.append(bert_sim)
            
            # SBERT similarity
            sbert_sim = float(cosine_similarity(
                self.get_sbert_embedding(student_desc).reshape(1, -1),
                self.get_sbert_embedding(proj_type).reshape(1, -1)
            )[0][0])
            sbert_scores.append(sbert_sim)
        
        # Combined semantic score
        bert_semantic_score = np.mean(bert_scores) if bert_scores else 0.5
        sbert_semantic_score = np.mean(sbert_scores) if sbert_scores else 0.5
        semantic_score = 0.4 * bert_semantic_score + 0.6 * sbert_semantic_score
        
        # Final combined score
        return 0.5 * direct_score + 0.5 * semantic_score

    def normalize_scores(self, matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize scores across all matches"""
        if not matches:
            return matches
            
        score_matrices = {
            'research': [],
            'methodology': [],
            'technical': [],
            'domain': [],
            'project': []
        }
        
        for match in matches:
            scores = match['detailed_scores']
            score_matrices['research'].append([scores['research_alignment']])
            score_matrices['methodology'].append([scores['methodology_match']])
            score_matrices['technical'].append([scores['technical_skills']])
            score_matrices['domain'].append([scores['domain_knowledge']])
            score_matrices['project'].append([scores['project_type_match']])
        
        for key in score_matrices:
            if score_matrices[key]:
                normalized_scores = self.scalers[key].fit_transform(score_matrices[key])
                for i, match in enumerate(matches):
                    match['detailed_scores'][f'{key}_alignment'] = float(normalized_scores[i][0])
        
        return matches

    def match_supervisors(self, student_data: Dict[str, Any], 
                         supervisors: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Match students with supervisors using multiple criteria"""
        results = []
        
        for supervisor in supervisors:
            research_score = self.calculate_research_alignment(
                student_data['project_description'],
                supervisor['interests']
            )
            
            methodology_score = self.calculate_methodology_match(
                student_data['project_description'],
                supervisor['interests']
            )
            
            student_skills = self.extract_technical_skills(student_data['project_description'])
            supervisor_skills = self.extract_technical_skills(supervisor['interests'])
            
            technical_score = 0.0
            if student_skills and supervisor_skills:
                category_weights = {
                    'programming': 0.25,
                    'machine_learning': 0.30,
                    'data_analysis': 0.25,
                    'cloud': 0.20
                }
                
                for category in self.technical_skills:
                    student_category = student_skills.get(category, {})
                    supervisor_category = supervisor_skills.get(category, {})
                    
                    if student_category and supervisor_category:
                        subcategory_matches = 0
                        total_subcategories = 0
                        
                        for subcategory in set(student_category.keys()) | set(supervisor_category.keys()):
                            student_skills_set = set(student_category.get(subcategory, []))
                            supervisor_skills_set = set(supervisor_category.get(subcategory, []))
                            
                            if student_skills_set and supervisor_skills_set:
                                match_ratio = len(student_skills_set & supervisor_skills_set) / len(student_skills_set | supervisor_skills_set)
                                subcategory_matches += match_ratio
                                total_subcategories += 1
                        
                        if total_subcategories > 0:
                            category_score = subcategory_matches / total_subcategories
                            technical_score += category_weights[category] * category_score
            else:
                technical_score = 0.5

            domain_score = self.calculate_domain_knowledge(
                student_data['project_description'],
                supervisor['interests']
            )
            
            project_type_score = self.calculate_project_type_match(
                student_data['project_description'],
                supervisor.get('project_types', [])
            )
            
            detailed_scores = {
                'research_alignment': research_score,
                'methodology_match': methodology_score,
                'technical_skills': technical_score,
                'domain_knowledge': domain_score,
                'project_type_match': project_type_score
            }
            
            matching_skills = {}
            for category in self.technical_skills:
                student_category = student_skills.get(category, {})
                supervisor_category = supervisor_skills.get(category, {})
                
                category_matches = {}
                for subcategory in set(student_category.keys()) & set(supervisor_category.keys()):
                    student_skills_set = set(student_category.get(subcategory, []))
                    supervisor_skills_set = set(supervisor_category.get(subcategory, []))
                    matches = student_skills_set & supervisor_skills_set
                    if matches:
                        category_matches[subcategory] = list(matches)
                
                if category_matches:
                    matching_skills[category] = category_matches
            
            confidence_score = self._calculate_confidence_score(
                student_data['project_description'],
                supervisor['interests'],
                matching_skills
            )
            
            final_score = sum(
                self.domain_weights[weight] * detailed_scores[score]
                for weight, score in {
                    'research_alignment': 'research_alignment',
                    'methodology_match': 'methodology_match',
                    'technical_skills': 'technical_skills',
                    'domain_knowledge': 'domain_knowledge',
                    'project_type_match': 'project_type_match'
                }.items()
            )
            
            final_score *= confidence_score
            
            results.append({
                'supervisor_name': supervisor['name'],
                'final_score': final_score,
                'confidence_score': confidence_score,
                'detailed_scores': detailed_scores,
                'matching_skills': matching_skills,
                'methodology_overlap': self._extract_methodology(supervisor['interests']),
                'research_keywords': self._extract_research_keywords(
                    student_data['project_description'],
                    supervisor['interests']
                )
            })
        
        normalized_results = self.normalize_scores(results)
        return sorted(normalized_results, key=lambda x: x['final_score'], reverse=True)

    def _calculate_confidence_score(self, student_desc: str, supervisor_interests: str, 
                                 matching_skills: Dict) -> float:
        """Calculate confidence score"""
        confidence = 1.0
        
        min_desc_length = 50
        if len(student_desc) < min_desc_length or len(supervisor_interests) < min_desc_length:
            confidence *= 0.8
        
        if not matching_skills:
            confidence *= 0.9
        
        if not self._extract_methodology(student_desc) or not self._extract_methodology(supervisor_interests):
            confidence *= 0.9
        
        keywords = self._extract_research_keywords(student_desc, supervisor_interests)
        if len(keywords) < 3:
            confidence *= 0.9
        
        return confidence

    def _extract_research_keywords(self, student_desc: str, supervisor_interests: str) -> List[str]:
        """Extract matching research keywords"""
        tfidf_matrix = self.tfidf.fit_transform([student_desc, supervisor_interests])
        feature_names = self.tfidf.get_feature_names_out()
        
        student_terms = set(feature_names[i] for i in tfidf_matrix[0].nonzero()[1])
        supervisor_terms = set(feature_names[i] for i in tfidf_matrix[1].nonzero()[1])
        
        return list(student_terms & supervisor_terms)


# Visualization and reporting functions could be added here

def visualize_results(matches: List[Dict[str, Any]], output_file: str = 'matching_results.png'):
    """Create enhanced visualization of matching results with multiple plots"""
    plt.style.use('seaborn')
    fig = plt.figure(figsize=(15, 10))
    
    # Create grid for multiple plots
    gs = plt.GridSpec(2, 2, figure=fig)
    
    # 1. Main stacked bar chart (top left)
    ax1 = fig.add_subplot(gs[0, :])
    supervisors = [m['supervisor_name'] for m in matches[:5]]  # Top 5 matches
    scores = [m['detailed_scores'] for m in matches[:5]]
    
    bottom = np.zeros(len(supervisors))
    colors = ['#4B79BF', '#5DAE8B', '#466983', '#7B90A5', '#2E5073']
    
    for i, (category, weight) in enumerate(DOMAIN_WEIGHTS.items()):
        values = [s[category] * weight for s in scores]
        ax1.bar(supervisors, values, bottom=bottom, label=category.replace('_', ' ').title(),
                color=colors[i], alpha=0.8)
        bottom += values
    
    ax1.set_title('Top 5 Matches - Score Breakdown', pad=20)
    ax1.set_xlabel('Supervisors')
    ax1.set_ylabel('Weighted Score')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Confidence scores (bottom left)
    ax2 = fig.add_subplot(gs[1, 0])
    confidence_scores = [m['confidence_score'] for m in matches[:5]]
    bars = ax2.bar(supervisors, confidence_scores, color='#2E5073', alpha=0.7)
    ax2.set_title('Match Confidence Scores')
    ax2.set_xlabel('Supervisors')
    ax2.set_ylabel('Confidence Score')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom')
    
    # 3. Skills matching heatmap (bottom right)
    ax3 = fig.add_subplot(gs[1, 1])
    skill_categories = list(DOMAIN_WEIGHTS.keys())
    skill_scores = np.array([[m['detailed_scores'][cat] for cat in skill_categories] 
                            for m in matches[:3]])  # Top 3 matches
    
    im = ax3.imshow(skill_scores, cmap='YlOrRd', aspect='auto')
    ax3.set_xticks(np.arange(len(skill_categories)))
    ax3.set_yticks(np.arange(len(matches[:3])))
    ax3.set_xticklabels([cat.replace('_', ' ').title() for cat in skill_categories])
    ax3.set_yticklabels([m['supervisor_name'] for m in matches[:3]])
    plt.setp(ax3.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax3.set_title('Detailed Score Comparison')
    
    # Add colorbar
    cbar = fig.colorbar(im)
    cbar.set_label('Score', rotation=270, labelpad=15)
    
    # Add score values in cells
    for i in range(len(matches[:3])):
        for j in range(len(skill_categories)):
            text = ax3.text(j, i, f'{skill_scores[i, j]:.2f}',
                          ha="center", va="center", color="black")
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def generate_report(student_data: Dict[str, Any], matches: List[Dict[str, Any]]) -> str:
    """Generate enhanced detailed matching report"""
    report = [
        f"Matching Report for {student_data['student_name']}",
        "=" * 50,
        "\nProject Description:",
        student_data['project_description'].strip(),
        "\nTop Matches Analysis",
        "=" * 50
    ]
    
    for i, match in enumerate(matches[:5], 1):  # Top 5 matches
        report.extend([
            f"\n{i}. {match['supervisor_name']}",
            f"Overall Match Score: {match['final_score']:.3f}",
            f"Confidence Score: {match['confidence_score']:.3f}",
            "\nDetailed Scores:",
            f"- Research Alignment: {match['detailed_scores']['research_alignment']:.3f}",
            f"- Methodology Match: {match['detailed_scores']['methodology_match']:.3f}",
            f"- Technical Skills: {match['detailed_scores']['technical_skills']:.3f}",
            f"- Domain Knowledge: {match['detailed_scores']['domain_knowledge']:.3f}",
            f"- Project Type Match: {match['detailed_scores']['project_type_match']:.3f}",
            "\nMatching Skills Analysis:"
        ])
        
        # Add detailed skills breakdown
        if match['matching_skills']:
            for category, subcategories in match['matching_skills'].items():
                report.append(f"\n{category.replace('_', ' ').title()}:")
                for subcategory, skills in subcategories.items():
                    report.append(f"  - {subcategory}: {', '.join(skills)}")
        else:
            report.append("  No direct skill matches found")
        
        # Add methodology analysis
        if match['methodology_overlap']:
            report.append("\nMethodology Compatibility:")
            for method_type, approaches in match['methodology_overlap'].items():
                report.append(f"  - {method_type.replace('_', ' ').title()}: {', '.join(approaches)}")
        
        # Add research keywords
        if match.get('research_keywords'):
            report.append("\nShared Research Keywords:")
            report.append("  " + ", ".join(match['research_keywords']))
        
        report.append("\n" + "-" * 50)
    
    return "\n".join(report)