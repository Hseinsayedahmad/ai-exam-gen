import streamlit as st
import PyPDF2
import io
import json
import sqlite3
import smtplib
import secrets
import hashlib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import random
import re
from datetime import datetime, timedelta
import base64
import os
import time
import requests
import string

# For PDF generation
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    PDF_GENERATION_AVAILABLE = True
except ImportError:
    PDF_GENERATION_AVAILABLE = False

# Email Configuration
EMAIL_CONFIG = {
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    "email": "hseinsa57@gmail.com",  # 🔴 UPDATE WITH YOUR GMAIL
    "password": "htvc jjbc wfyu nlhz",  # 🔴 UPDATE WITH YOUR APP PASSWORD
    "from_name": "AI Exam Generator Support"
}

# Google Gemini API Configuration
GEMINI_CONFIG = {
    "api_key": "AIzaSyDwBxI6aJGrEk_RyJLyob9ytLyJAaQYkyo",  # Your working API key
    "model": "gemini-1.5-flash",
    "base_url": "https://generativelanguage.googleapis.com/v1beta/models",
    "max_tokens": 1000,
    "temperature": 0.7
}

# Configuration and Data Models
class QuestionType(Enum):
    MULTIPLE_CHOICE = "multiple_choice"
    TRUE_FALSE = "true_false"
    SHORT_ANSWER = "short_answer"
    ESSAY = "essay"
    FILL_IN_BLANK = "fill_in_blank"
    MATCHING = "matching"

class Difficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

@dataclass
class ExamSettings:
    question_types: List[QuestionType]
    difficulty: Difficulty
    num_questions: int
    time_limit: Optional[int]
    topics: List[str]
    exam_title: str = "Generated Exam"
    course_name: str = "Course"

@dataclass
class Question:
    question_text: str
    question_type: QuestionType
    difficulty: Difficulty
    options: Optional[List[str]] = None
    correct_answer: str = ""
    explanation: str = ""
    topic: str = ""

@dataclass
class ChatMessage:
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime

# Database setup
def init_database():
    """Initialize the user database"""
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            plain_password TEXT,
            is_verified BOOLEAN DEFAULT TRUE,
            verification_token TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            verified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Add default admin user if not exists
    admin_hash = hashlib.sha256("6368".encode()).hexdigest()
    cursor.execute('''
        INSERT OR IGNORE INTO users (username, email, password_hash, plain_password, is_verified, verified_at)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', ("Hsein", "hsein@example.com", admin_hash, "6368", True, datetime.now()))
    
    conn.commit()
    conn.close()

class PDFProcessor:
    """Enhanced PDF processing with content analysis"""
    
    @staticmethod
    def extract_text_from_pdf(pdf_file) -> str:
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            return ""
    
    @staticmethod
    def preprocess_text(text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        return text.strip()
    
    @staticmethod
    def extract_topics(text: str) -> List[str]:
        topics = []
        lines = text.split('\n')
        for line in lines:
            if len(line) < 100 and any(word in line.lower() for word in ['chapter', 'section', 'topic', 'introduction', 'overview', 'summary']):
                topics.append(line.strip())
        return topics[:10]
    
    @staticmethod
    def extract_key_concepts(text: str) -> List[str]:
        """Extract key concepts and terms from the text"""
        patterns = [
            r'(\w+) is defined as',
            r'(\w+) refers to',
            r'(\w+) means',
            r'(\w+) can be described as',
            r'The term (\w+)',
            r'(\w+) is a type of',
            r'(\w+) is an example of'
        ]
        
        concepts = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            concepts.extend([match.lower() for match in matches if len(match) > 3])
        
        return list(set(concepts))[:20]
    
    @staticmethod
    def extract_relationships(text: str) -> List[Tuple[str, str, str]]:
        """Extract cause-effect and other relationships from text"""
        relationships = []
        
        cause_patterns = [
            r'(.+) causes (.+)',
            r'(.+) results in (.+)',
            r'(.+) leads to (.+)',
            r'Because of (.+), (.+)',
            r'Due to (.+), (.+)'
        ]
        
        for pattern in cause_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match) == 2:
                    cause = match[0].strip()[:50]
                    effect = match[1].strip()[:50]
                    if len(cause) > 5 and len(effect) > 5:
                        relationships.append((cause, "causes", effect))
        
        return relationships[:10]

class AdvancedQuestionGenerator:
    """Advanced logic-based question generator that creates meaningful questions"""
    
    def __init__(self):
        self.concepts = []
        self.relationships = []
        self.topics = []
        self.key_sentences = []
    
    def analyze_content(self, content: str) -> None:
        """Analyze content to extract meaningful elements"""
        self.concepts = PDFProcessor.extract_key_concepts(content)
        self.relationships = PDFProcessor.extract_relationships(content)
        self.topics = PDFProcessor.extract_topics(content)
        
        sentences = content.split('.')
        self.key_sentences = [
            s.strip() for s in sentences 
            if 50 < len(s.strip()) < 200 and 
            any(word in s.lower() for word in ['is', 'are', 'can', 'will', 'must', 'should', 'because', 'therefore'])
        ][:30]
    
    def generate_questions(self, content: str, settings: ExamSettings) -> List[Question]:
        """Generate intelligent questions based on content analysis"""
        self.analyze_content(content)
        questions = []
        
        if not self.key_sentences and not self.concepts:
            # Fallback to simple generation if no meaningful content found
            return self._generate_fallback_questions(content, settings)
        
        questions_per_type = max(1, settings.num_questions // len(settings.question_types))
        
        for question_type in settings.question_types:
            type_questions = []
            
            if question_type == QuestionType.MULTIPLE_CHOICE:
                type_questions = self._generate_multiple_choice(questions_per_type, settings.difficulty)
            elif question_type == QuestionType.TRUE_FALSE:
                type_questions = self._generate_true_false(questions_per_type, settings.difficulty)
            elif question_type == QuestionType.FILL_IN_BLANK:
                type_questions = self._generate_fill_blank(questions_per_type, settings.difficulty)
            elif question_type == QuestionType.SHORT_ANSWER:
                type_questions = self._generate_short_answer(questions_per_type, settings.difficulty)
            elif question_type == QuestionType.ESSAY:
                type_questions = self._generate_essay(questions_per_type, settings.difficulty)
            
            questions.extend(type_questions)
        
        random.shuffle(questions)
        return questions[:settings.num_questions]
    
    def _generate_fallback_questions(self, content: str, settings: ExamSettings) -> List[Question]:
        """Generate basic questions when advanced analysis fails"""
        questions = []
        sentences = [s.strip() for s in content.split('.') if len(s.strip()) > 20][:20]
        
        for i, sentence in enumerate(sentences):
            if len(questions) >= settings.num_questions:
                break
                
            question_type = random.choice(settings.question_types)
            
            if question_type == QuestionType.FILL_IN_BLANK:
                words = sentence.split()
                if len(words) > 5:
                    blank_index = len(words) // 2
                    correct_word = words[blank_index]
                    words[blank_index] = "______"
                    question_text = " ".join(words) + "?"
                    
                    questions.append(Question(
                        question_text=question_text,
                        question_type=question_type,
                        difficulty=settings.difficulty,
                        correct_answer=correct_word.strip('.,!?'),
                        explanation=f"The missing word is '{correct_word}'",
                        topic="General Knowledge"
                    ))
            
            elif question_type == QuestionType.TRUE_FALSE:
                questions.append(Question(
                    question_text=f"True or False: {sentence}",
                    question_type=question_type,
                    difficulty=settings.difficulty,
                    correct_answer="True",
                    explanation="Based on the course material",
                    topic="Comprehension"
                ))
            
            elif question_type == QuestionType.SHORT_ANSWER:
                questions.append(Question(
                    question_text=f"Explain the following statement: {sentence[:50]}...",
                    question_type=question_type,
                    difficulty=settings.difficulty,
                    correct_answer="Answer should explain the statement based on course content",
                    explanation="Look for key concepts and explanations",
                    topic="Analysis"
                ))
        
        return questions
    
    def _generate_multiple_choice(self, num_questions: int, difficulty: Difficulty) -> List[Question]:
        """Generate logical multiple choice questions"""
        questions = []
        
        for _ in range(num_questions):
            if not self.concepts and not self.key_sentences:
                break
                
            if self.concepts and random.choice([True, False]):
                question = self._create_concept_multiple_choice(difficulty)
            else:
                question = self._create_comprehension_multiple_choice(difficulty)
            
            if question:
                questions.append(question)
        
        return questions
    
    def _create_concept_multiple_choice(self, difficulty: Difficulty) -> Optional[Question]:
        """Create a concept-based multiple choice question"""
        if not self.concepts:
            return None
        
        concept = random.choice(self.concepts).title()
        
        concept_sentence = None
        for sentence in self.key_sentences:
            if concept.lower() in sentence.lower():
                concept_sentence = sentence
                break
        
        if not concept_sentence:
            return None
        
        if difficulty == Difficulty.EASY:
            question_text = f"What is {concept}?"
            correct_answer = f"As described in the text about {concept.lower()}"
        elif difficulty == Difficulty.MEDIUM:
            question_text = f"According to the text, {concept} is best described as:"
            correct_answer = f"The concept explained in the course material"
        else:
            question_text = f"How does {concept} relate to the main concepts discussed in the text?"
            correct_answer = f"It connects to the key principles as outlined"
        
        distractors = [
            "A concept not discussed in this material",
            "An outdated theory no longer relevant",
            "A simple definition without context"
        ]
        
        options = distractors + [correct_answer]
        random.shuffle(options)
        
        formatted_options = [f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)]
        correct_letter = chr(65 + options.index(correct_answer))
        
        return Question(
            question_text=question_text,
            question_type=QuestionType.MULTIPLE_CHOICE,
            difficulty=difficulty,
            options=formatted_options,
            correct_answer=f"{correct_letter}. {correct_answer}",
            explanation=f"Based on the discussion of {concept} in the source material",
            topic=concept
        )
    
    def _create_comprehension_multiple_choice(self, difficulty: Difficulty) -> Optional[Question]:
        """Create a comprehension-based multiple choice question"""
        if not self.key_sentences:
            return None
        
        source_sentence = random.choice(self.key_sentences)
        words = source_sentence.split()
        
        if len(words) < 8:
            return None
        
        if "because" in source_sentence.lower():
            parts = source_sentence.lower().split("because")
            if len(parts) == 2:
                effect = parts[0].strip()
                cause = parts[1].strip()
                question_text = f"According to the text, what is the primary reason for {effect[:30]}...?"
                correct_answer = f"Because {cause[:40]}..."
            else:
                question_text = f"What does the text suggest about: {source_sentence[:50]}...?"
                correct_answer = "The explanation provided in the source material"
        else:
            key_phrase = " ".join(words[len(words)//3:len(words)*2//3])
            question_text = f"The text discusses {key_phrase}. What is the main point?"
            correct_answer = "The key concept as explained in the material"
        
        distractors = [
            "A common misconception about the topic",
            "An oversimplified explanation",
            "A perspective not supported by the text"
        ]
        
        options = distractors + [correct_answer]
        random.shuffle(options)
        
        formatted_options = [f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)]
        correct_letter = chr(65 + options.index(correct_answer))
        
        return Question(
            question_text=question_text,
            question_type=QuestionType.MULTIPLE_CHOICE,
            difficulty=difficulty,
            options=formatted_options,
            correct_answer=f"{correct_letter}. {correct_answer}",
            explanation="Based on comprehension of the source material",
            topic="Reading Comprehension"
        )
    
    def _generate_true_false(self, num_questions: int, difficulty: Difficulty) -> List[Question]:
        """Generate logical true/false questions"""
        questions = []
        
        for _ in range(num_questions):
            if not self.key_sentences:
                break
            
            sentence = random.choice(self.key_sentences)
            
            if difficulty == Difficulty.EASY:
                question_text = f"True or False: {sentence}"
                correct_answer = "True"
                explanation = "This statement is directly supported by the text"
            elif difficulty == Difficulty.MEDIUM:
                modified = self._modify_sentence_for_false(sentence)
                question_text = f"True or False: {modified}"
                correct_answer = "False"
                explanation = "This statement contains modifications not supported by the original text"
            else:
                if "because" in sentence.lower() or "therefore" in sentence.lower():
                    question_text = f"True or False: Based on the text, we can conclude that {sentence[:60]}..."
                    correct_answer = "True"
                    explanation = "This conclusion can be logically inferred from the text"
                else:
                    question_text = f"True or False: {sentence}"
                    correct_answer = "True"
                    explanation = "This statement is supported by the source material"
            
            questions.append(Question(
                question_text=question_text,
                question_type=QuestionType.TRUE_FALSE,
                difficulty=difficulty,
                correct_answer=correct_answer,
                explanation=explanation,
                topic="Comprehension"
            ))
        
        return questions
    
    def _modify_sentence_for_false(self, sentence: str) -> str:
        """Modify a sentence to make it false but plausible"""
        modifications = [
            (r'\bincreases\b', 'decreases'),
            (r'\bimproves\b', 'reduces'),
            (r'\ballows\b', 'prevents'),
            (r'\benables\b', 'inhibits'),
            (r'\bpositive\b', 'negative'),
            (r'\beffective\b', 'ineffective'),
            (r'\balways\b', 'never'),
            (r'\ball\b', 'no'),
            (r'\bmust\b', 'cannot'),
            (r'\bcan\b', 'cannot')
        ]
        
        for pattern, replacement in modifications:
            if re.search(pattern, sentence, re.IGNORECASE):
                return re.sub(pattern, replacement, sentence, count=1, flags=re.IGNORECASE)
        
        if sentence.startswith("The"):
            return sentence.replace("The", "The concept that", 1) + " is not accurate"
        return f"It is incorrect that {sentence.lower()}"
    
    def _generate_fill_blank(self, num_questions: int, difficulty: Difficulty) -> List[Question]:
        """Generate intelligent fill-in-the-blank questions"""
        questions = []
        
        for _ in range(num_questions):
            if not self.key_sentences and not self.concepts:
                break
            
            sentence = random.choice(self.key_sentences)
            words = sentence.split()
            
            if len(words) < 6:
                continue
            
            if difficulty == Difficulty.EASY:
                target_words = [w for w in words if len(w) > 6 and w.lower() not in ['the', 'and', 'or', 'but']]
            elif difficulty == Difficulty.MEDIUM:
                target_words = [w for w in words if w.lower() in self.concepts or len(w) > 5]
            else:
                target_words = [w for w in words if w.lower() in ['because', 'therefore', 'however', 'although', 'since']]
                if not target_words:
                    target_words = [w for w in words if len(w) > 4]
            
            if not target_words:
                continue
            
            target_word = random.choice(target_words)
            blank_sentence = sentence.replace(target_word, "______", 1)
            
            questions.append(Question(
                question_text=f"Fill in the blank: {blank_sentence}",
                question_type=QuestionType.FILL_IN_BLANK,
                difficulty=difficulty,
                correct_answer=target_word.strip('.,!?'),
                explanation=f"The missing word '{target_word}' is key to understanding this concept",
                topic="Vocabulary"
            ))
        
        return questions
    
    def _generate_short_answer(self, num_questions: int, difficulty: Difficulty) -> List[Question]:
        """Generate thoughtful short answer questions"""
        questions = []
        
        question_starters = {
            Difficulty.EASY: ["What is", "Define", "List the main"],
            Difficulty.MEDIUM: ["Explain how", "Describe the relationship between", "What are the key differences between"],
            Difficulty.HARD: ["Analyze the implications of", "Evaluate the effectiveness of", "Critically assess"]
        }
        
        for _ in range(num_questions):
            if not self.concepts and not self.topics:
                break
            
            if self.concepts:
                concept = random.choice(self.concepts)
                starter = random.choice(question_starters[difficulty])
                question_text = f"{starter} {concept} as discussed in the material."
                topic = concept.title()
            else:
                topic = random.choice(self.topics) if self.topics else "General"
                starter = random.choice(question_starters[difficulty])
                question_text = f"{starter} points related to {topic}."
            
            questions.append(Question(
                question_text=question_text,
                question_type=QuestionType.SHORT_ANSWER,
                difficulty=difficulty,
                correct_answer=f"Answer should demonstrate understanding based on course material",
                explanation=f"Look for key points from the text",
                topic=topic
            ))
        
        return questions
    
    def _generate_essay(self, num_questions: int, difficulty: Difficulty) -> List[Question]:
        """Generate comprehensive essay questions"""
        questions = []
        
        essay_prompts = {
            Difficulty.EASY: [
                "Summarize the main concepts discussed in the material.",
                "Explain the key topics covered in this text."
            ],
            Difficulty.MEDIUM: [
                "Compare and contrast the different approaches mentioned in the text.",
                "Discuss how the concepts in this material relate to each other."
            ],
            Difficulty.HARD: [
                "Critically analyze the arguments presented in the material.",
                "Evaluate the strengths and limitations of the theories discussed.",
                "Synthesize the information to propose new insights or applications."
            ]
        }
        
        for _ in range(min(num_questions, 2)):
            prompt = random.choice(essay_prompts[difficulty])
            
            questions.append(Question(
                question_text=prompt,
                question_type=QuestionType.ESSAY,
                difficulty=difficulty,
                correct_answer="Essay should demonstrate deep understanding and critical thinking",
                explanation="Look for comprehensive analysis, clear arguments, and evidence from the text",
                topic="Critical Analysis"
            ))
        
        return questions

class PDFExamGenerator:
    """Enhanced PDF generator with professional formatting"""
    
    def __init__(self):
        if not PDF_GENERATION_AVAILABLE:
            st.error("PDF generation requires reportlab. Install with: pip install reportlab")
    
    def create_exam_pdf(self, questions: List[Question], settings: ExamSettings) -> bytes:
        """Create a professionally formatted PDF exam"""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=1*inch)
        
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], 
                                   fontSize=20, spaceAfter=30, alignment=1, textColor=colors.darkblue)
        subtitle_style = ParagraphStyle('Subtitle', parent=styles['Heading2'], 
                                      fontSize=14, spaceAfter=20, alignment=1)
        question_style = ParagraphStyle('Question', parent=styles['Normal'], 
                                      fontSize=11, spaceAfter=12, leftIndent=0, fontName='Helvetica-Bold')
        option_style = ParagraphStyle('Option', parent=styles['Normal'], 
                                    fontSize=10, spaceAfter=6, leftIndent=20)
        instruction_style = ParagraphStyle('Instructions', parent=styles['Normal'], 
                                         fontSize=10, spaceAfter=8, leftIndent=10)
        
        story = []
        
        story.append(Paragraph(settings.exam_title, title_style))
        story.append(Paragraph(f"Course: {settings.course_name}", subtitle_style))
        story.append(Spacer(1, 0.3*inch))
        
        exam_info = f"""
        <b>Date:</b> {datetime.now().strftime('%B %d, %Y')}<br/>
        <b>Time Limit:</b> {settings.time_limit} minutes<br/>
        <b>Total Questions:</b> {len(questions)}<br/>
        <b>Total Points:</b> {len(questions) * 10} points
        """
        story.append(Paragraph(exam_info, styles['Normal']))
        story.append(Spacer(1, 0.4*inch))
        
        instructions = f"""
        <b>INSTRUCTIONS - READ CAREFULLY:</b><br/><br/>
        1. <b>Time Management:</b> You have {settings.time_limit} minutes to complete this exam<br/>
        2. <b>Multiple Choice:</b> Select the BEST answer for each question<br/>
        3. <b>True/False:</b> Mark T for True or F for False<br/>
        4. <b>Written Responses:</b> Write clearly and concisely<br/>
        5. <b>Review:</b> Check your answers before submitting<br/>
        6. <b>Academic Integrity:</b> This is an individual assessment<br/><br/>
        <b>Good luck!</b>
        """
        
        story.append(Paragraph(instructions, instruction_style))
        story.append(PageBreak())
        
        for i, question in enumerate(questions, 1):
            question_header = f"<b>Question {i}</b> ({question.difficulty.value.title()}) - 10 points"
            story.append(Paragraph(question_header, question_style))
            
            story.append(Paragraph(question.question_text, styles['Normal']))
            story.append(Spacer(1, 0.1*inch))
            
            if question.question_type == QuestionType.MULTIPLE_CHOICE and question.options:
                for option in question.options:
                    story.append(Paragraph(f"○ {option}", option_style))
                story.append(Spacer(1, 0.1*inch))
            
            elif question.question_type == QuestionType.TRUE_FALSE:
                story.append(Paragraph("○ True     ○ False", option_style))
                story.append(Spacer(1, 0.1*inch))
            
            elif question.question_type == QuestionType.FILL_IN_BLANK:
                story.append(Paragraph("Answer: ________________________________", option_style))
                story.append(Spacer(1, 0.1*inch))
            
            elif question.question_type == QuestionType.SHORT_ANSWER:
                story.append(Paragraph("<b>Answer:</b>", option_style))
                for _ in range(4):
                    story.append(Paragraph("_" * 85, styles['Normal']))
                story.append(Spacer(1, 0.1*inch))
            
            elif question.question_type == QuestionType.ESSAY:
                story.append(Paragraph("<b>Essay Response:</b> (Use additional paper if needed)", option_style))
                for _ in range(8):
                    story.append(Paragraph("_" * 85, styles['Normal']))
                story.append(Spacer(1, 0.1*inch))
            
            story.append(Spacer(1, 0.2*inch))
            
            if i % 4 == 0 and i < len(questions):
                story.append(PageBreak())
        
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
    
    def create_answer_key_pdf(self, questions: List[Question], settings: ExamSettings) -> bytes:
        """Create a comprehensive answer key PDF"""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=1*inch)
        
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], 
                                   fontSize=18, spaceAfter=30, alignment=1)
        
        story = []
        
        story.append(Paragraph(f"{settings.exam_title} - Answer Key", title_style))
        story.append(Paragraph(f"Course: {settings.course_name}", styles['Heading2']))
        story.append(Spacer(1, 0.5*inch))
        
        for i, question in enumerate(questions, 1):
            story.append(Paragraph(f"<b>Question {i}:</b> {question.question_text}", 
                                 styles['Normal']))
            story.append(Paragraph(f"<b>Correct Answer:</b> {question.correct_answer}", 
                                 styles['Normal']))
            if question.explanation:
                story.append(Paragraph(f"<b>Explanation:</b> {question.explanation}", 
                                     styles['Normal']))
            story.append(Paragraph(f"<b>Difficulty:</b> {question.difficulty.value.title()}", 
                                 styles['Normal']))
            story.append(Paragraph(f"<b>Topic:</b> {question.topic}", 
                                 styles['Normal']))
            story.append(Spacer(1, 0.2*inch))
        
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()

class GeminiManager:
    """Manages Google Gemini API interactions for PDF chatbot"""
    
    def __init__(self):
        self.api_key = GEMINI_CONFIG["api_key"]
        self.model = GEMINI_CONFIG["model"]
        self.base_url = GEMINI_CONFIG["base_url"]
        self.max_tokens = GEMINI_CONFIG["max_tokens"]
        self.temperature = GEMINI_CONFIG["temperature"]
    
    def check_api_status(self) -> Tuple[bool, str]:
        """Check if Gemini API is accessible and working"""
        if self.api_key == "your_gemini_api_key_here":
            return False, "Please configure your Gemini API key"
        
        try:
            url = f"{self.base_url}/{self.model}:generateContent"
            headers = {"Content-Type": "application/json"}
            
            test_payload = {
                "contents": [{"parts": [{"text": "Hello, please respond with 'API Working'"}]}],
                "generationConfig": {"maxOutputTokens": 10, "temperature": 0.1}
            }
            
            response = requests.post(f"{url}?key={self.api_key}", headers=headers, json=test_payload, timeout=10)
            
            if response.status_code == 200:
                return True, "✅ Gemini API connected successfully!"
            elif response.status_code == 400:
                error_detail = response.json().get('error', {}).get('message', 'Unknown error')
                return False, f"API Error: {error_detail}"
            elif response.status_code == 403:
                return False, "Invalid API key or quota exceeded"
            else:
                return False, f"API responded with status {response.status_code}"
                
        except requests.exceptions.ConnectionError:
            return False, "No internet connection or API unreachable"
        except requests.exceptions.Timeout:
            return False, "API request timed out"
        except Exception as e:
            return False, f"Error checking API: {str(e)}"
    
    def get_api_usage_info(self) -> Dict[str, str]:
        """Get information about API usage and limits"""
        return {
            "model": self.model,
            "daily_limit": "1,500 requests/day",
            "monthly_limit": "1M tokens/month",
            "rate_limit": "15 requests/minute",
            "cost": "FREE"
        }
    
    def chat_with_pdf(self, pdf_content: str, user_message: str, chat_history: List[ChatMessage]) -> str:
        """Send a chat message with PDF context to Gemini API"""
        try:
            if self.api_key == "your_gemini_api_key_here":
                return "❌ Please configure your Gemini API key in the code. Check the sidebar for instructions."
            
            system_context = f"""You are a helpful AI assistant that answers questions based on the provided PDF document content. 

DOCUMENT CONTENT:
{pdf_content[:6000]}...

INSTRUCTIONS:
- Answer questions based ONLY on the provided document content
- If a question cannot be answered from the document, politely say so
- Be concise but thorough in your explanations
- Use examples from the document when relevant
- If asked to summarize, focus on the key points from the document"""

            conversation_parts = [{"text": system_context}]
            
            recent_history = chat_history[-6:] if len(chat_history) > 6 else chat_history
            for msg in recent_history:
                role_prefix = "Human" if msg.role == "user" else "Assistant"
                conversation_parts.append({"text": f"{role_prefix}: {msg.content}"})
            
            conversation_parts.append({"text": f"Human: {user_message}"})
            conversation_parts.append({"text": "Assistant:"})
            
            url = f"{self.base_url}/{self.model}:generateContent"
            headers = {"Content-Type": "application/json"}
            
            payload = {
                "contents": [{"parts": conversation_parts}],
                "generationConfig": {
                    "maxOutputTokens": self.max_tokens,
                    "temperature": self.temperature,
                    "topP": 0.8,
                    "topK": 40
                },
                "safetySettings": [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
                ]
            }
            
            response = requests.post(f"{url}?key={self.api_key}", headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                
                if 'candidates' in result and len(result['candidates']) > 0:
                    candidate = result['candidates'][0]
                    if 'content' in candidate and 'parts' in candidate['content']:
                        response_text = candidate['content']['parts'][0].get('text', '')
                        return response_text
                    else:
                        return "❌ No response generated. Please try rephrasing your question."
                else:
                    return "❌ No response candidates found. Please try a different question."
            
            elif response.status_code == 429:
                return "⚠️ Rate limit exceeded. Please wait a moment before asking another question."
            elif response.status_code == 400:
                error_detail = response.json().get('error', {}).get('message', 'Unknown error')
                return f"❌ Request error: {error_detail}"
            elif response.status_code == 403:
                return "❌ API access denied. Please check your API key and quota."
            else:
                return f"❌ API error (Status {response.status_code}). Please try again."
                
        except requests.exceptions.Timeout:
            return "⏰ Request timed out. Please try asking a shorter or simpler question."
        except requests.exceptions.ConnectionError:
            return "🌐 No internet connection. Please check your network and try again."
        except Exception as e:
            return f"❌ Unexpected error: {str(e)}"

class UserManager:
    """User management with credential emailing"""
    
    @staticmethod
    def hash_password(password: str) -> str:
        return hashlib.sha256(password.encode()).hexdigest()
    
    @staticmethod
    def user_exists(username: str, email: str) -> Dict[str, bool]:
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        
        cursor.execute("SELECT username FROM users WHERE username = ?", (username,))
        username_exists = cursor.fetchone() is not None
        
        cursor.execute("SELECT email FROM users WHERE email = ?", (email,))
        email_exists = cursor.fetchone() is not None
        
        conn.close()
        return {"username": username_exists, "email": email_exists}
    
    @staticmethod
    def create_user(username: str, email: str, password: str) -> Tuple[bool, str]:
        try:
            exists = UserManager.user_exists(username, email)
            if exists["username"]:
                return False, "Username already exists"
            if exists["email"]:
                return False, "Email already registered"
            
            password_hash = UserManager.hash_password(password)
            verification_token = secrets.token_urlsafe(32)
            
            conn = sqlite3.connect('users.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO users (username, email, password_hash, plain_password, verification_token, is_verified, verified_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (username, email, password_hash, password, verification_token, True, datetime.now()))
            
            conn.commit()
            conn.close()
            
            if EMAIL_CONFIG["email"] != "your_email@gmail.com":
                if UserManager.send_credentials_email(email, username, password):
                    return True, "Account created successfully! Check your email for login credentials."
                else:
                    return True, "Account created! Email sending failed - your credentials are saved."
            else:
                return True, "Account created! Email not configured - save your credentials."
                
        except Exception as e:
            return False, f"Error creating account: {str(e)}"
    
    @staticmethod
    def send_credentials_email(email: str, username: str, password: str) -> bool:
        try:
            login_url = "http://localhost:8501"
            
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>Your AI Exam Generator Account</title>
            </head>
            <body style="font-family: Arial, sans-serif; background-color: #f4f4f4; margin: 0; padding: 0;">
                <div style="max-width: 600px; margin: 0 auto; background-color: white;">
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 40px 20px; text-align: center;">
                        <h1 style="color: white; margin: 0; font-size: 2.5em;">🎓</h1>
                        <h2 style="color: white; margin: 10px 0 0 0;">AI Exam Generator</h2>
                        <p style="color: rgba(255,255,255,0.9); margin: 5px 0 0 0;">Your Account is Ready!</p>
                    </div>
                    
                    <div style="padding: 40px 30px;">
                        <h2 style="color: #333;">🎉 Welcome to AI Exam Generator!</h2>
                        
                        <div style="background: #f8f9fa; border: 2px solid #e9ecef; border-radius: 10px; padding: 25px; margin: 30px 0;">
                            <h3 style="color: #333; text-align: center;">🔐 Your Login Credentials</h3>
                            <p><strong>Username:</strong> {username}</p>
                            <p><strong>Password:</strong> {password}</p>
                            
                            <div style="text-align: center; margin-top: 25px;">
                                <a href="{login_url}" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px 30px; text-decoration: none; border-radius: 8px; font-weight: bold;">
                                    🚀 Login to Your Account
                                </a>
                            </div>
                        </div>
                        
                        <h3>✨ What You Can Do:</h3>
                        <ul>
                            <li>🖥️ <strong>Interactive Exams:</strong> Take exams online with instant feedback</li>
                            <li>📄 <strong>PDF Exam Generation:</strong> Create professional exam PDFs</li>
                            <li>🤖 <strong>PDF Chatbot:</strong> Chat with your PDFs using Google Gemini AI</li>
                        </ul>
                    </div>
                </div>
            </body>
            </html>
            """
            
            msg = MIMEMultipart('alternative')
            msg['Subject'] = "🎓 Welcome to AI Exam Generator - Your Login Credentials"
            msg['From'] = f"{EMAIL_CONFIG['from_name']} <{EMAIL_CONFIG['email']}>"
            msg['To'] = email
            
            html_part = MIMEText(html_content, 'html')
            msg.attach(html_part)
            
            server = smtplib.SMTP(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port'])
            server.starttls()
            server.login(EMAIL_CONFIG['email'], EMAIL_CONFIG['password'])
            server.send_message(msg)
            server.quit()
            
            return True
            
        except Exception as e:
            st.error(f"Email sending failed: {str(e)}")
            return False
    
    @staticmethod
    def authenticate_user(username: str, password: str) -> Tuple[bool, str, bool]:
        try:
            conn = sqlite3.connect('users.db')
            cursor = conn.cursor()
            
            password_hash = UserManager.hash_password(password)
            
            cursor.execute('''
                SELECT username, is_verified FROM users 
                WHERE username = ? AND password_hash = ?
            ''', (username, password_hash))
            
            result = cursor.fetchone()
            conn.close()
            
            if not result:
                return False, "Invalid username or password", False
            
            username, is_verified = result
            return True, f"Welcome back, {username}!", True
            
        except Exception as e:
            return False, f"Login error: {str(e)}", False

class AuthenticationManager:
    """Handles session management"""
    
    @staticmethod
    def is_authenticated() -> bool:
        return st.session_state.get('authenticated', False)
    
    @staticmethod
    def get_current_user() -> str:
        return st.session_state.get('username', '')
    
    @staticmethod
    def login(username: str) -> None:
        st.session_state['authenticated'] = True
        st.session_state['username'] = username
        st.session_state['login_time'] = datetime.now()
    
    @staticmethod
    def logout() -> None:
        for key in ['authenticated', 'username', 'login_time']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

def signup_page():
    """Signup page"""
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="color: #1f2937; font-size: 2.5rem; margin-bottom: 0.5rem;">🎓 Join AI Exam Generator</h1>
        <p style="color: #6b7280; font-size: 1.1rem;">Create your account and receive login credentials via email</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        with st.form("signup_form"):
            st.markdown("### 📝 Create Account")
            
            username = st.text_input("Username *", placeholder="Choose a unique username")
            email = st.text_input("Email Address *", placeholder="your.email@example.com")
            password = st.text_input("Password *", type="password", placeholder="Create a strong password")
            confirm_password = st.text_input("Confirm Password *", type="password", placeholder="Re-enter your password")
            
            st.info("""
            📧 **What happens after signup:**
            - Account created instantly with access to all features
            - Beautiful email sent with your username and password
            - Access to Interactive Exams, PDF Generation, and Google Gemini AI Chatbot
            """)
            
            terms_accepted = st.checkbox("I agree to the Terms of Service and Privacy Policy")
            
            col_signup, col_back = st.columns([1, 1])
            
            with col_signup:
                signup_button = st.form_submit_button("🚀 Create Account & Send Credentials", type="primary", use_container_width=True)
            
            with col_back:
                back_button = st.form_submit_button("← Back to Login", use_container_width=True)
        
        if signup_button:
            errors = []
            
            if not username or len(username) < 3:
                errors.append("Username must be at least 3 characters")
            if not email or "@" not in email or "." not in email:
                errors.append("Please enter a valid email address")
            if not password or len(password) < 6:
                errors.append("Password must be at least 6 characters")
            if password != confirm_password:
                errors.append("Passwords do not match")
            if not terms_accepted:
                errors.append("Please accept the Terms of Service")
            
            if errors:
                for error in errors:
                    st.error(f"❌ {error}")
            else:
                with st.spinner("Creating account and sending credentials email..."):
                    success, message = UserManager.create_user(username, email, password)
                
                if success:
                    st.success(f"✅ {message}")
                    st.balloons()
                    
                    st.markdown(f"""
                    ### 🎉 Account Created Successfully!
                    
                    **Username:** `{username}`  
                    **Email:** `{email}`
                    
                    📧 **Check your email** for login credentials and quick start guide!
                    """)
                    
                    if st.button("🔑 Go to Login Page", type="primary"):
                        st.session_state['show_signup'] = False
                        st.rerun()
                else:
                    st.error(f"❌ {message}")
        
        if back_button:
            st.session_state['show_signup'] = False
            st.rerun()

def login_page():
    """Login page"""
    if st.session_state.get('show_signup', False):
        signup_page()
        return
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="color: #1f2937; font-size: 2.5rem; margin-bottom: 0.5rem;">🎓 AI Exam Generator</h1>
        <p style="color: #6b7280; font-size: 1.1rem;">Please sign in to continue</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        with st.form("login_form"):
            st.markdown("### 🔐 Sign In")
            
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            
            col_login, col_signup = st.columns([1, 1])
            
            with col_login:
                login_button = st.form_submit_button("🚀 Sign In", type="primary", use_container_width=True)
            
            with col_signup:
                signup_button = st.form_submit_button("📝 Create Account", use_container_width=True)
        
        if login_button:
            if username and password:
                success, message, verified = UserManager.authenticate_user(username, password)
                
                if success and verified:
                    AuthenticationManager.login(username)
                    st.success(message)
                    st.balloons()
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(message)
            else:
                st.warning("⚠️ Please enter both username and password")
        
        if signup_button:
            st.session_state['show_signup'] = True
            st.rerun()
        
        with st.expander("📋 Access Information"):
            st.info("""
            ### 🔧 Admin Access
            **Username:** `Hsein`  
            **Password:** `6368`
            
            ### 🆕 Features Available:
            - 🖥️ **Interactive Exams:** Complete online exam system
            - 📄 **PDF Generation:** Professional exam creation with answer keys
            - 🤖 **PDF Chatbot:** Chat with documents using Google Gemini AI
            """)

def interactive_exam_page():
    """COMPLETE Interactive exam page with FIXED answer persistence"""
    st.header("🖥️ Interactive AI Exam Generator")
    st.markdown("Upload a PDF and take a comprehensive interactive exam with instant feedback")
    
    # Initialize session state for interactive exam
    if 'interactive_questions' not in st.session_state:
        st.session_state.interactive_questions = []
    if 'interactive_exam_started' not in st.session_state:
        st.session_state.interactive_exam_started = False
    if 'interactive_user_answers' not in st.session_state:
        st.session_state.interactive_user_answers = {}
    if 'exam_start_time' not in st.session_state:
        st.session_state.exam_start_time = None
    if 'time_remaining' not in st.session_state:
        st.session_state.time_remaining = None
    
    # Sidebar settings
    with st.sidebar:
        st.header("⚙️ Interactive Exam Settings")
        
        # Question types
        st.subheader("Question Types")
        question_types = []
        for q_type in QuestionType:
            default_value = q_type in [QuestionType.MULTIPLE_CHOICE, QuestionType.TRUE_FALSE]
            if st.checkbox(q_type.value.replace('_', ' ').title(), 
                          value=default_value, key=f"int_{q_type.value}"):
                question_types.append(q_type)
        
        difficulty = st.selectbox("Difficulty Level", options=[d.value.title() for d in Difficulty], index=1)
        num_questions = st.slider("Number of Questions", 1, 25, 8)
        time_limit = st.number_input("Time Limit (minutes)", 5, 120, 20)
        
        # Display exam status
        if st.session_state.interactive_exam_started:
            if st.session_state.time_remaining is not None:
                mins, secs = divmod(st.session_state.time_remaining, 60)
                st.error(f"⏱️ Time Remaining: {int(mins):02d}:{int(secs):02d}")
            
            answered = len(st.session_state.interactive_user_answers)
            total = len(st.session_state.interactive_questions)
            st.info(f"📝 Progress: {answered}/{total} questions answered")
    
    # Main content
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📄 Upload Course Material")
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file", 
            type="pdf", 
            key="interactive_pdf",
            help="Upload your course material to generate an interactive exam"
        )
        
        if uploaded_file:
            with st.spinner("Processing PDF for interactive exam..."):
                text_content = PDFProcessor.extract_text_from_pdf(uploaded_file)
                clean_content = PDFProcessor.preprocess_text(text_content)
                topics = PDFProcessor.extract_topics(clean_content)
            
            st.success(f"✅ PDF processed! Found {len(topics)} topics")
            
            # Show content stats
            st.info(f"""
            📊 **Content Analysis:**
            - Words: {len(clean_content.split()):,}
            - Characters: {len(clean_content):,}
            - Topics: {len(topics)}
            """)
            
            if topics:
                st.subheader("📚 Detected Topics")
                for topic in topics[:5]:
                    st.write(f"• {topic}")
                
                if len(topics) > 5:
                    with st.expander(f"Show {len(topics) - 5} more topics"):
                        for topic in topics[5:]:
                            st.write(f"• {topic}")
            
            if st.button("🚀 Generate Interactive Exam", type="primary", use_container_width=True):
                if not question_types:
                    st.error("Please select at least one question type!")
                else:
                    settings = ExamSettings(
                        question_types=question_types,
                        difficulty=Difficulty(difficulty.lower()),
                        num_questions=num_questions,
                        time_limit=time_limit,
                        topics=topics,
                        exam_title="Interactive Exam",
                        course_name="Course Material"
                    )
                    
                    with st.spinner("Generating intelligent exam questions..."):
                        generator = AdvancedQuestionGenerator()
                        questions = generator.generate_questions(clean_content, settings)
                        
                        if questions:
                            st.session_state.interactive_questions = questions
                            st.session_state.interactive_exam_started = True
                            st.session_state.interactive_user_answers = {}
                            st.session_state.exam_start_time = datetime.now()
                            st.session_state.time_remaining = time_limit * 60  # Convert to seconds
                            
                            st.success(f"✅ Generated {len(questions)} questions!")
                            st.balloons()
                            st.rerun()
                        else:
                            st.error("Failed to generate questions. Please try with different settings.")
    
    with col2:
        if st.session_state.interactive_exam_started and st.session_state.interactive_questions:
            st.subheader("📝 Interactive Exam")
            
            # Timer logic
            if st.session_state.time_remaining is not None and st.session_state.time_remaining > 0:
                st.session_state.time_remaining -= 1
                if st.session_state.time_remaining <= 0:
                    st.error("⏰ Time's up! Exam auto-submitted.")
            
            # FIXED Display questions with proper answer persistence
            for i, question in enumerate(st.session_state.interactive_questions):
                with st.expander(
                    f"Question {i+1} - {question.difficulty.value.title()} ({question.question_type.value.replace('_', ' ').title()})", 
                    expanded=(i < 3)  # Expand first 3 questions
                ):
                    st.markdown(f"**{question.question_text}**")
                    
                    # Get previously selected answer from session state
                    previous_answer = st.session_state.interactive_user_answers.get(i, None)
                    
                    # Question type specific UI with proper state management
                    if question.question_type == QuestionType.MULTIPLE_CHOICE and question.options:
                        # Find the index of previously selected answer
                        default_index = None
                        if previous_answer:
                            try:
                                default_index = question.options.index(previous_answer)
                            except ValueError:
                                default_index = None
                        
                        answer = st.radio(
                            "Choose your answer:",
                            question.options,
                            key=f"interactive_q_{i}",
                            index=default_index
                        )
                        if answer:
                            st.session_state.interactive_user_answers[i] = answer
                    
                    elif question.question_type == QuestionType.TRUE_FALSE:
                        # Find the index of previously selected answer
                        options = ["True", "False"]
                        default_index = None
                        if previous_answer in options:
                            default_index = options.index(previous_answer)
                        
                        answer = st.radio(
                            "Choose your answer:",
                            options,
                            key=f"interactive_q_{i}",
                            index=default_index
                        )
                        if answer:
                            st.session_state.interactive_user_answers[i] = answer
                    
                    elif question.question_type == QuestionType.FILL_IN_BLANK:
                        # Use previous answer as default value
                        answer = st.text_input(
                            "Fill in the blank:",
                            key=f"interactive_q_{i}",
                            value=previous_answer if previous_answer else "",
                            placeholder="Type your answer..."
                        )
                        if answer.strip():  # Only store non-empty answers
                            st.session_state.interactive_user_answers[i] = answer.strip()
                        elif i in st.session_state.interactive_user_answers and not answer.strip():
                            # Remove empty answers from session state
                            del st.session_state.interactive_user_answers[i]
                    
                    elif question.question_type == QuestionType.SHORT_ANSWER:
                        # Use previous answer as default value
                        answer = st.text_area(
                            "Your answer:",
                            key=f"interactive_q_{i}",
                            value=previous_answer if previous_answer else "",
                            placeholder="Write your answer here...",
                            height=100
                        )
                        if answer.strip():  # Only store non-empty answers
                            st.session_state.interactive_user_answers[i] = answer.strip()
                        elif i in st.session_state.interactive_user_answers and not answer.strip():
                            # Remove empty answers from session state
                            del st.session_state.interactive_user_answers[i]
                    
                    elif question.question_type == QuestionType.ESSAY:
                        # Use previous answer as default value
                        answer = st.text_area(
                            "Essay response:",
                            key=f"interactive_q_{i}",
                            value=previous_answer if previous_answer else "",
                            placeholder="Write your detailed essay response here...",
                            height=150
                        )
                        if answer.strip():  # Only store non-empty answers
                            st.session_state.interactive_user_answers[i] = answer.strip()
                        elif i in st.session_state.interactive_user_answers and not answer.strip():
                            # Remove empty answers from session state
                            del st.session_state.interactive_user_answers[i]
                    
                    # Show question metadata
                    col_topic, col_diff = st.columns(2)
                    with col_topic:
                        if question.topic:
                            st.caption(f"📚 Topic: {question.topic}")
                    with col_diff:
                        difficulty_color = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}
                        st.caption(f"{difficulty_color.get(question.difficulty.value, '⚪')} {question.difficulty.value.title()}")
                    
                    # Show if question is answered
                    if i in st.session_state.interactive_user_answers:
                        st.success("✅ Answered")
                    else:
                        st.warning("⏳ Not answered yet")
            
            # Submit exam - FIXED VERSION
            st.markdown("---")
            col_submit, col_reset = st.columns([2, 1])
            
            with col_submit:
                if st.button("✅ Submit Interactive Exam", type="primary", use_container_width=True):
                    # Calculate results with FIXED answer comparison
                    correct = 0
                    total = len(st.session_state.interactive_questions)
                    detailed_results = []
                    
                    for i, question in enumerate(st.session_state.interactive_questions):
                        user_answer = st.session_state.interactive_user_answers.get(i, "")
                        is_correct = False
                        
                        # FIXED: Better answer comparison logic
                        if question.question_type == QuestionType.MULTIPLE_CHOICE:
                            # Handle both "A. Answer" format and just "Answer" format
                            correct_answer = question.correct_answer
                            
                            # Extract just the answer part if it has letter prefix
                            if '. ' in correct_answer:
                                correct_text = correct_answer.split('. ', 1)[1]
                            else:
                                correct_text = correct_answer
                            
                            # Extract just the answer part from user selection
                            if '. ' in user_answer:
                                user_text = user_answer.split('. ', 1)[1]
                            else:
                                user_text = user_answer
                            
                            # Compare the actual text content
                            is_correct = user_text.lower().strip() == correct_text.lower().strip()
                            
                            # Alternative: Compare the full selected option
                            if not is_correct:
                                is_correct = user_answer.lower().strip() == correct_answer.lower().strip()
                        
                        elif question.question_type == QuestionType.TRUE_FALSE:
                            # Simple true/false comparison
                            correct_answer = question.correct_answer.lower().strip()
                            user_text = user_answer.lower().strip()
                            is_correct = user_text == correct_answer
                        
                        elif question.question_type == QuestionType.FILL_IN_BLANK:
                            # Remove punctuation for comparison
                            correct_clean = question.correct_answer.lower().strip().translate(str.maketrans('', '', string.punctuation))
                            user_clean = user_answer.lower().strip().translate(str.maketrans('', '', string.punctuation))
                            is_correct = correct_clean == user_clean
                        
                        elif question.question_type in [QuestionType.SHORT_ANSWER, QuestionType.ESSAY]:
                            # For subjective questions, check if there's a substantial answer
                            is_correct = len(user_answer.strip()) > 10
                        
                        else:
                            # Default case
                            is_correct = user_answer.lower().strip() == question.correct_answer.lower().strip()
                        
                        if is_correct:
                            correct += 1
                        
                        detailed_results.append({
                            'question': question.question_text,
                            'user_answer': user_answer,
                            'correct_answer': question.correct_answer,
                            'is_correct': is_correct,
                            'explanation': question.explanation,
                            'topic': question.topic
                        })
                    
                    percentage = (correct / total) * 100 if total > 0 else 0
                    
                    # Show results
                    st.balloons()
                    st.success("🎉 Exam completed successfully!")
                    
                    # Results summary
                    st.subheader("📊 Your Results")
                    col_a, col_b, col_c = st.columns(3)
                    
                    with col_a:
                        st.metric("Score", f"{correct}/{total}")
                    with col_b:
                        st.metric("Percentage", f"{percentage:.1f}%")
                    with col_c:
                        if percentage >= 90:
                            grade, color = "A", "🟢"
                        elif percentage >= 80:
                            grade, color = "B", "🔵"
                        elif percentage >= 70:
                            grade, color = "C", "🟡"
                        elif percentage >= 60:
                            grade, color = "D", "🟠"
                        else:
                            grade, color = "F", "🔴"
                        st.metric("Grade", f"{color} {grade}")
                    
                    # Detailed results
                    with st.expander("📋 Detailed Question Review", expanded=True):
                        for i, result in enumerate(detailed_results):
                            st.markdown(f"**Question {i+1}:** {result['question']}")
                            
                            col_ans, col_corr = st.columns(2)
                            with col_ans:
                                status = "✅ Correct" if result['is_correct'] else "❌ Incorrect"
                                st.write(f"{status}")
                                st.write(f"**Your answer:** {result['user_answer'] or 'No answer provided'}")
                            with col_corr:
                                st.write(f"**Correct answer:** {result['correct_answer']}")
                            
                            if result['explanation']:
                                st.info(f"💡 **Explanation:** {result['explanation']}")
                            
                            if result['topic']:
                                st.caption(f"📚 **Topic:** {result['topic']}")
                            
                            st.markdown("---")
                    
                    # Performance insights
                    st.subheader("📈 Performance Insights")
                    
                    # By question type
                    type_performance = {}
                    for i, result in enumerate(detailed_results):
                        q_type = st.session_state.interactive_questions[i].question_type.value.replace('_', ' ').title()
                        if q_type not in type_performance:
                            type_performance[q_type] = {'correct': 0, 'total': 0}
                        type_performance[q_type]['total'] += 1
                        if result['is_correct']:
                            type_performance[q_type]['correct'] += 1
                    
                    if type_performance:
                        st.write("**Performance by Question Type:**")
                        for q_type, performance in type_performance.items():
                            type_percentage = (performance['correct'] / performance['total']) * 100
                            st.write(f"• {q_type}: {performance['correct']}/{performance['total']} ({type_percentage:.0f}%)")
                    
                    # By topic
                    topic_performance = {}
                    for result in detailed_results:
                        topic = result['topic'] or 'General'
                        if topic not in topic_performance:
                            topic_performance[topic] = {'correct': 0, 'total': 0}
                        topic_performance[topic]['total'] += 1
                        if result['is_correct']:
                            topic_performance[topic]['correct'] += 1
                    
                    if topic_performance:
                        st.write("**Performance by Topic:**")
                        for topic, performance in topic_performance.items():
                            topic_percentage = (performance['correct'] / performance['total']) * 100
                            st.write(f"• {topic}: {performance['correct']}/{performance['total']} ({topic_percentage:.0f}%)")
                    
                    # Study recommendations
                    incorrect_topics = [result['topic'] for result in detailed_results if not result['is_correct'] and result['topic']]
                    if incorrect_topics:
                        st.subheader("📚 Study Recommendations")
                        unique_topics = list(set(incorrect_topics))
                        st.write("Focus on these topics for improvement:")
                        for topic in unique_topics:
                            st.write(f"• {topic}")
                    
                    # Reset exam option
                    st.markdown("---")
                    if st.button("🔄 Take Another Exam", type="secondary", use_container_width=True):
                        st.session_state.interactive_exam_started = False
                        st.session_state.interactive_questions = []
                        st.session_state.interactive_user_answers = {}
                        st.session_state.exam_start_time = None
                        st.session_state.time_remaining = None
                        st.rerun()
            
            with col_reset:
                if st.button("🔄 Reset Exam", type="secondary", use_container_width=True):
                    st.session_state.interactive_exam_started = False
                    st.session_state.interactive_questions = []
                    st.session_state.interactive_user_answers = {}
                    st.session_state.exam_start_time = None
                    st.session_state.time_remaining = None
                    st.rerun()
        
        elif not uploaded_file:
            st.info("👈 Upload a PDF file to get started with interactive exams")
        else:
            st.info("Configure your exam settings and click 'Generate Interactive Exam'")

def advanced_pdf_page():
    """COMPLETE Advanced PDF generation page with full functionality"""
    st.header("📄 Advanced PDF Exam Generator")
    st.markdown("Create professional, downloadable exam PDFs with intelligent question generation")
    
    if not PDF_GENERATION_AVAILABLE:
        st.error("⚠️ PDF generation requires the reportlab library")
        st.code("pip install reportlab")
        st.stop()
    
    st.info("""
    🎯 **Professional PDF Generation Features:**
    - **Smart Question Creation**: AI-powered question generation with logical reasoning
    - **Multiple Formats**: Various question types with proper formatting
    - **Answer Keys**: Comprehensive answer sheets with explanations
    - **Print Ready**: Professional layout suitable for classroom use
    """)
    
    # Settings in main area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📄 Course Material")
        uploaded_file = st.file_uploader("Upload course PDF", type="pdf", key="pdf_gen")
        
        if uploaded_file:
            with st.spinner("🔍 Analyzing PDF content for exam generation..."):
                text_content = PDFProcessor.extract_text_from_pdf(uploaded_file)
                clean_content = PDFProcessor.preprocess_text(text_content)
                topics = PDFProcessor.extract_topics(clean_content)
                concepts = PDFProcessor.extract_key_concepts(clean_content)
                relationships = PDFProcessor.extract_relationships(clean_content)
            
            st.success(f"✅ Content analysis complete!")
            
            # Show analysis results
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Words", f"{len(clean_content.split()):,}")
            with col_b:
                st.metric("Key Concepts", len(concepts))
            with col_c:
                st.metric("Relationships", len(relationships))
            
            if topics:
                st.subheader("📚 Detected Topics")
                for topic in topics[:5]:
                    st.write(f"• {topic}")
                
                if len(topics) > 5:
                    with st.expander(f"Show all {len(topics)} topics"):
                        for topic in topics:
                            st.write(f"• {topic}")
            
            if concepts:
                st.subheader("🔑 Key Concepts Found")
                concept_display = ", ".join(concepts[:8])
                st.write(concept_display)
                if len(concepts) > 8:
                    with st.expander(f"Show all {len(concepts)} concepts"):
                        for concept in concepts:
                            st.write(f"• {concept}")
            
            if relationships:
                st.subheader("🔗 Cause-Effect Relationships")
                for i, (cause, relation, effect) in enumerate(relationships[:3]):
                    st.write(f"• {cause} → {effect}")
                
                if len(relationships) > 3:
                    with st.expander(f"Show all {len(relationships)} relationships"):
                        for cause, relation, effect in relationships:
                            st.write(f"• {cause} → {effect}")
    
    with col2:
        st.subheader("⚙️ Professional Exam Configuration")
        
        # Exam details
        exam_title = st.text_input("Exam Title", value="Final Examination")
        course_name = st.text_input("Course Name", value="Course 101")
        
        # Question intelligence level
        st.subheader("🧠 Question Generation")
        intelligence_level = st.selectbox(
            "Generation Method",
            ["Advanced AI Logic", "Rule-Based", "Hybrid Approach"],
            index=0,
            help="Advanced AI Logic creates more intelligent and contextual questions"
        )
        
        # Question types
        st.subheader("Question Types")
        question_types = []
        
        col_a, col_b = st.columns(2)
        with col_a:
            if st.checkbox("Multiple Choice", value=True, help="Intelligent distractors based on content", key="pdf_mc"):
                question_types.append(QuestionType.MULTIPLE_CHOICE)
            if st.checkbox("True/False", help="Content-based with logical modifications", key="pdf_tf"):
                question_types.append(QuestionType.TRUE_FALSE)
            if st.checkbox("Fill in Blank", help="Key vocabulary and concepts", key="pdf_fib"):
                question_types.append(QuestionType.FILL_IN_BLANK)
        
        with col_b:
            if st.checkbox("Short Answer", help="Concept explanation questions", key="pdf_sa"):
                question_types.append(QuestionType.SHORT_ANSWER)
            if st.checkbox("Essay", help="Critical thinking and analysis", key="pdf_essay"):
                question_types.append(QuestionType.ESSAY)
        
        # Advanced settings
        difficulty = st.selectbox("Difficulty Level", ["Easy", "Medium", "Hard"], index=1)
        num_questions = st.slider("Number of Questions", 5, 40, 15)
        time_limit = st.number_input("Time Limit (minutes)", 30, 300, 90)
        
        # Show expected question distribution
        if question_types:
            st.subheader("📊 Question Distribution")
            per_type = max(1, num_questions // len(question_types))
            remaining = num_questions % len(question_types)
            
            for i, q_type in enumerate(question_types):
                count = per_type + (1 if i < remaining else 0)
                st.write(f"• {q_type.value.replace('_', ' ').title()}: {count} questions")
    
    # Generate button
    if uploaded_file and question_types:
        if st.button("🚀 Generate Professional Exam PDF", type="primary", use_container_width=True):
            with st.spinner("🧠 Creating professional exam with intelligent questions..."):
                
                # Create settings
                settings = ExamSettings(
                    question_types=question_types,
                    difficulty=Difficulty(difficulty.lower()),
                    num_questions=num_questions,
                    time_limit=time_limit,
                    topics=topics,
                    exam_title=exam_title,
                    course_name=course_name
                )
                
                # Generate questions using advanced generator
                generator = AdvancedQuestionGenerator()
                questions = generator.generate_questions(clean_content, settings)
                
                if questions:
                    # Create PDF generator
                    pdf_gen = PDFExamGenerator()
                    
                    # Generate both exam and answer key PDFs
                    exam_pdf = pdf_gen.create_exam_pdf(questions, settings)
                    answer_key_pdf = pdf_gen.create_answer_key_pdf(questions, settings)
                    
                    st.success(f"✅ Generated professional exam with {len(questions)} intelligent questions!")
                    
                    # Download buttons
                    col_d1, col_d2 = st.columns(2)
                    
                    with col_d1:
                        st.download_button(
                            label="📥 Download Exam PDF",
                            data=exam_pdf,
                            file_name=f"{exam_title.replace(' ', '_')}_exam.pdf",
                            mime="application/pdf",
                            type="primary",
                            use_container_width=True
                        )
                    
                    with col_d2:
                        st.download_button(
                            label="🔑 Download Answer Key",
                            data=answer_key_pdf,
                            file_name=f"{exam_title.replace(' ', '_')}_answer_key.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                    
                    # Enhanced preview
                    st.subheader("📋 Exam Preview")
                    
                    preview_count = min(4, len(questions))
                    for i, question in enumerate(questions[:preview_count], 1):
                        with st.expander(f"Preview Question {i} - {question.question_type.value.replace('_', ' ').title()} ({question.difficulty.value})"):
                            st.write(f"**Question:** {question.question_text}")
                            
                            if question.options:
                                st.write("**Options:**")
                                for option in question.options:
                                    st.write(f"  {option}")
                            
                            st.write(f"**Correct Answer:** {question.correct_answer}")
                            
                            if question.explanation:
                                st.write(f"**Explanation:** {question.explanation}")
                            
                            if question.topic:
                                st.write(f"**Topic:** {question.topic}")
                    
                    if len(questions) > preview_count:
                        st.write(f"... and {len(questions) - preview_count} more professionally crafted questions")
                    
                    # Quality metrics
                    st.subheader("📈 Question Quality Analysis")
                    col_q1, col_q2, col_q3, col_q4 = st.columns(4)
                    
                    with col_q1:
                        concept_questions = sum(1 for q in questions if any(concept in q.question_text.lower() for concept in concepts))
                        st.metric("Concept-Based", f"{concept_questions}/{len(questions)}")
                    
                    with col_q2:
                        difficulty_counts = {d.value: sum(1 for q in questions if q.difficulty.value == d.value) for d in Difficulty}
                        is_balanced = max(difficulty_counts.values()) - min(difficulty_counts.values()) <= 2
                        st.metric("Difficulty Balance", "✓ Balanced" if is_balanced else "⚠ Uneven")
                    
                    with col_q3:
                        unique_topics = len(set(q.topic for q in questions if q.topic))
                        st.metric("Topic Coverage", f"{unique_topics} topics")
                    
                    with col_q4:
                        has_explanations = sum(1 for q in questions if q.explanation)
                        st.metric("With Explanations", f"{has_explanations}/{len(questions)}")
                    
                    # Additional exam statistics
                    st.subheader("📊 Exam Statistics")
                    
                    # Question type breakdown
                    type_counts = {}
                    for question in questions:
                        q_type = question.question_type.value.replace('_', ' ').title()
                        type_counts[q_type] = type_counts.get(q_type, 0) + 1
                    
                    st.write("**Question Type Distribution:**")
                    for q_type, count in type_counts.items():
                        percentage = (count / len(questions)) * 100
                        st.write(f"• {q_type}: {count} questions ({percentage:.1f}%)")
                    
                    # Estimated completion time
                    estimated_time = {
                        QuestionType.MULTIPLE_CHOICE: 1.5,
                        QuestionType.TRUE_FALSE: 1.0,
                        QuestionType.FILL_IN_BLANK: 1.0,
                        QuestionType.SHORT_ANSWER: 3.0,
                        QuestionType.ESSAY: 8.0
                    }
                    
                    total_estimated = sum(estimated_time.get(q.question_type, 2.0) for q in questions)
                    st.info(f"⏱️ **Estimated completion time:** {total_estimated:.0f} minutes (Set limit: {time_limit} minutes)")
                
                else:
                    st.error("Failed to generate questions. Please try with different settings or a different PDF.")
    
    elif uploaded_file and not question_types:
        st.warning("⚠️ Please select at least one question type to generate the exam")
    elif not uploaded_file:
        st.info("👆 Upload a PDF file to begin creating professional exams")

def pdf_chatbot_page():
    """COMPLETE PDF Chatbot page with Google Gemini API"""
    st.header("🤖 PDF Chatbot with Google Gemini")
    st.markdown("Upload a PDF and have intelligent conversations with your documents using Google's advanced AI")
    
    # Initialize session state for chatbot
    if 'chatbot_messages' not in st.session_state:
        st.session_state.chatbot_messages = []
    if 'pdf_content' not in st.session_state:
        st.session_state.pdf_content = ""
    if 'pdf_uploaded' not in st.session_state:
        st.session_state.pdf_uploaded = False
    
    # Initialize Gemini manager
    gemini = GeminiManager()
    
    # Sidebar for Gemini configuration
    with st.sidebar:
        st.header("🤖 Google Gemini AI")
        
        # Check API status
        with st.spinner("Checking Gemini API status..."):
            is_connected, status_message = gemini.check_api_status()
        
        if is_connected:
            st.success(f"✅ {status_message}")
            
            # API usage info
            usage_info = gemini.get_api_usage_info()
            st.info(f"""
            **Model:** {usage_info['model']}  
            **Daily Limit:** {usage_info['daily_limit']}  
            **Monthly Limit:** {usage_info['monthly_limit']}  
            **Rate Limit:** {usage_info['rate_limit']}  
            **Cost:** {usage_info['cost']} 🎉
            """)
            
        else:
            st.error(f"❌ {status_message}")
            
            if GEMINI_CONFIG["api_key"] == "your_gemini_api_key_here":
                st.markdown("""
                ### 🔑 Get Your FREE Gemini API Key:
                
                1. **Visit:** [Google AI Studio](https://makersuite.google.com/app/apikey)
                2. **Click:** "Create API Key"
                3. **Copy** the key
                4. **Update** the code:
                
                ```python
                GEMINI_CONFIG = {
                    "api_key": "your_actual_api_key_here"
                }
                ```
                
                ### ✨ Why Gemini API is Perfect:
                - 🆓 **FREE** with generous limits
                - ⚡ **No downloads** (unlike Ollama)
                - 🌐 **Works with any internet speed**
                - 🚀 **Setup in 30 seconds**
                """)
        
        # Chat controls
        st.markdown("---")
        st.subheader("💬 Chat Controls")
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chatbot_messages = []
            st.rerun()
        
        if st.session_state.pdf_uploaded:
            st.success(f"📄 PDF loaded ({len(st.session_state.pdf_content)} characters)")
            
            # Chat statistics
            if st.session_state.chatbot_messages:
                user_msgs = [msg for msg in st.session_state.chatbot_messages if msg.role == "user"]
                ai_msgs = [msg for msg in st.session_state.chatbot_messages if msg.role == "assistant"]
                
                st.markdown("### 📊 Chat Stats")
                st.write(f"• Total messages: {len(st.session_state.chatbot_messages)}")
                st.write(f"• Your questions: {len(user_msgs)}")
                st.write(f"• AI responses: {len(ai_msgs)}")
        else:
            st.warning("📄 No PDF uploaded")
    
    # Main content area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📄 Upload Document")
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file to chat with",
            type="pdf",
            key="chatbot_pdf",
            help="Upload any PDF document to start intelligent conversations with Google Gemini AI"
        )
        
        if uploaded_file:
            if not st.session_state.pdf_uploaded or st.session_state.get('last_uploaded_file') != uploaded_file.name:
                with st.spinner("Processing PDF for intelligent chat..."):
                    pdf_text = PDFProcessor.extract_text_from_pdf(uploaded_file)
                    clean_text = PDFProcessor.preprocess_text(pdf_text)
                    
                    if clean_text:
                        st.session_state.pdf_content = clean_text
                        st.session_state.pdf_uploaded = True
                        st.session_state.last_uploaded_file = uploaded_file.name
                        
                        # Clear previous chat when new PDF is uploaded
                        st.session_state.chatbot_messages = []
                        
                        st.success(f"✅ PDF processed successfully!")
                        st.info(f"""
                        📊 **Document Stats:**
                        - **Words:** {len(clean_text.split()):,}
                        - **Characters:** {len(clean_text):,}
                        - **Pages:** Approximately {len(clean_text) // 2000}
                        """)
                        
                        # Show PDF preview
                        with st.expander("📖 Document Preview"):
                            st.text_area("First 500 characters:", clean_text[:500], height=150, disabled=True)
                    else:
                        st.error("Failed to extract text from PDF. Please try a different file.")
        
        # Document Analysis
        if st.session_state.pdf_uploaded:
            st.subheader("📊 Document Analysis")
            topics = PDFProcessor.extract_topics(st.session_state.pdf_content)
            
            if topics:
                st.write("**Main Topics Found:**")
                for topic in topics[:5]:
                    st.write(f"• {topic}")
                
                if len(topics) > 5:
                    with st.expander(f"Show all {len(topics)} topics"):
                        for topic in topics:
                            st.write(f"• {topic}")
            
            # Smart suggested questions
            st.subheader("💡 Smart Question Suggestions")
            
            # Generate context-aware suggestions
            if topics:
                contextual_questions = [
                    f"What does this document say about {topics[0].lower()}?",
                    "Summarize the main points covered",
                    "What are the key concepts I should understand?",
                    "Create study questions from this content",
                    "Explain the most important topic in simple terms"
                ]
            else:
                contextual_questions = [
                    "What is this document about?",
                    "Summarize the main points",
                    "What are the key concepts?",
                    "Create study questions from this content",
                    "Explain the most important information"
                ]
            
            for question in contextual_questions:
                if st.button(f"💬 {question}", key=f"suggest_{hash(question)}", use_container_width=True):
                    if is_connected and st.session_state.pdf_uploaded:
                        # Add user message
                        st.session_state.chatbot_messages.append(
                            ChatMessage("user", question, datetime.now())
                        )
                        st.rerun()
    
    with col2:
        st.subheader("💬 Intelligent PDF Chat")
        
        if not st.session_state.pdf_uploaded:
            st.info("👈 Please upload a PDF to start intelligent conversations")
            
            # Show example conversation
            st.markdown("""
            ### 🎯 Example Conversation:
            
            **You:** "What is this document about?"  
            **AI:** "This document discusses machine learning fundamentals, covering supervised learning, neural networks, and practical applications..."
            
            **You:** "Explain neural networks in simple terms"  
            **AI:** "Neural networks are computing systems inspired by biological brains. They consist of interconnected nodes that process information..."
            
            **You:** "Create 5 study questions from this content"  
            **AI:** "Here are 5 study questions based on the document: 1) What are the main types of machine learning? 2) How do neural networks..."
            """)
            
        elif not is_connected:
            st.error("❌ Google Gemini API is not configured. Please check the sidebar for setup instructions.")
            
        else:
            # Chat interface
            chat_container = st.container()
            
            with chat_container:
                # Display chat messages
                for i, message in enumerate(st.session_state.chatbot_messages):
                    if message.role == "user":
                        with st.chat_message("user"):
                            st.write(message.content)
                            st.caption(f"📅 {message.timestamp.strftime('%H:%M:%S')}")
                    else:
                        with st.chat_message("assistant"):
                            st.write(message.content)
                            st.caption(f"🤖 Gemini • {message.timestamp.strftime('%H:%M:%S')}")
            
            # Chat input
            user_input = st.chat_input("Ask anything about your PDF...")
            
            if user_input:
                # Add user message
                st.session_state.chatbot_messages.append(
                    ChatMessage("user", user_input, datetime.now())
                )
                
                # Generate AI response
                with st.spinner("🤖 Gemini is analyzing your question..."):
                    try:
                        ai_response = gemini.chat_with_pdf(
                            st.session_state.pdf_content,
                            user_input,
                            st.session_state.chatbot_messages
                        )
                        
                        # Add AI response
                        st.session_state.chatbot_messages.append(
                            ChatMessage("assistant", ai_response, datetime.now())
                        )
                        
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")
                        st.session_state.chatbot_messages.append(
                            ChatMessage("assistant", f"Sorry, I encountered an error: {str(e)}", datetime.now())
                        )
                
                st.rerun()
            
            # Advanced chat features
            if st.session_state.chatbot_messages:
                st.markdown("---")
                
                # Chat analytics
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    st.metric("💬 Total Messages", len(st.session_state.chatbot_messages))
                
                with col_b:
                    user_messages = [msg for msg in st.session_state.chatbot_messages if msg.role == "user"]
                    st.metric("❓ Your Questions", len(user_messages))
                
                with col_c:
                    if st.session_state.chatbot_messages:
                        duration = st.session_state.chatbot_messages[-1].timestamp - st.session_state.chatbot_messages[0].timestamp
                        st.metric("⏱️ Chat Duration", f"{duration.seconds//60}m {duration.seconds%60}s")
                
                # Export chat option
                if st.button("📥 Export Chat History", use_container_width=True):
                    chat_export = []
                    for msg in st.session_state.chatbot_messages:
                        chat_export.append({
                            'timestamp': msg.timestamp.isoformat(),
                            'role': msg.role,
                            'content': msg.content
                        })
                    
                    chat_json = json.dumps(chat_export, indent=2)
                    st.download_button(
                        label="Download Chat as JSON",
                        data=chat_json,
                        file_name=f"pdf_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )

def main():
    """Main application with complete functionality for all three pages"""
    
    # Initialize database
    init_database()
    
    # Initialize session state
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False
    if 'show_signup' not in st.session_state:
        st.session_state['show_signup'] = False
    
    # Check authentication
    if not AuthenticationManager.is_authenticated():
        st.set_page_config(
            page_title="AI Exam Generator - Login",
            page_icon="🔐",
            layout="centered"
        )
        login_page()
        return
    
    # Main application (authenticated users only)
    st.set_page_config(
        page_title="AI Exam Generator",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header with user info and logout
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        st.title("🎓 AI Exam Generator")
    
    with col2:
        user = AuthenticationManager.get_current_user()
        login_time = st.session_state.get('login_time', datetime.now())
        st.success(f"👋 Welcome back, **{user}**!")
        st.caption(f"Logged in at {login_time.strftime('%H:%M:%S')}")
    
    with col3:
        if st.button("🚪 Logout", type="secondary"):
            AuthenticationManager.logout()
    
    st.markdown("---")
    
    # Sidebar navigation - THREE COMPLETE MODES
    st.sidebar.title(f"👤 {AuthenticationManager.get_current_user()}")
    page = st.sidebar.selectbox(
        "Choose Mode:",
        ["🖥️ Interactive Exam", "📄 PDF Exam Generator", "🤖 PDF Chatbot (Gemini)"]
    )
    
    # Route to appropriate page - ALL FULLY IMPLEMENTED
    if page == "🖥️ Interactive Exam":
        interactive_exam_page()
    elif page == "📄 PDF Exam Generator":
        advanced_pdf_page()
    else:  # PDF Chatbot with Gemini
        pdf_chatbot_page()

if __name__ == "__main__":
    main()