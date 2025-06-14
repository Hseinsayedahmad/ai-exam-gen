"""
Working PDF Test with Paper_project_Telecom_II.pdf
Uses your advanced rule-based model with fallback handling
"""

import PyPDF2
import os
import re
from collections import Counter
import random

# Simple fallback tokenizer if NLTK fails
def simple_tokenize(text):
    """Simple tokenization without NLTK"""
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    return sentences

def simple_pos_tag(words):
    """Simple POS tagging fallback"""
    # Basic heuristics for POS tagging
    tagged = []
    for word in words:
        if word.isupper() and len(word) > 1:
            tagged.append((word, 'NNP'))  # Proper noun
        elif word.endswith('ing'):
            tagged.append((word, 'VBG'))  # Verb
        elif word.endswith('ed'):
            tagged.append((word, 'VBD'))  # Past verb
        elif word.endswith('ly'):
            tagged.append((word, 'RB'))   # Adverb
        else:
            tagged.append((word, 'NN'))   # Default to noun
    return tagged

class SimplifiedQuestionGenerator:
    """Simplified version that works without NLTK downloads"""
    
    def __init__(self):
        self.stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        self.key_terms = []
        
    def analyze_text(self, text):
        """Simple text analysis"""
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        clean_words = [w for w in words if w not in self.stop_words]
        
        # Find key terms by frequency
        word_freq = Counter(clean_words)
        self.key_terms = [word for word, freq in word_freq.most_common(10)]
        
        return {
            'word_count': len(words),
            'key_terms': self.key_terms,
            'domain': self.classify_domain(clean_words)
        }
    
    def classify_domain(self, words):
        """Simple domain classification"""
        telecom_words = ['network', 'communication', 'signal', 'frequency', 'antenna', 'wireless', 'optical', 'fiber']
        cs_words = ['algorithm', 'computer', 'software', 'programming', 'data', 'system']
        
        telecom_score = sum(1 for word in words if word in telecom_words)
        cs_score = sum(1 for word in words if word in cs_words)
        
        if telecom_score > cs_score:
            return 'telecommunications'
        elif cs_score > 0:
            return 'computer_science'
        else:
            return 'general'
    
    def generate_questions(self, text, num_questions=5):
        """Generate questions with simplified approach"""
        analysis = self.analyze_text(text)
        sentences = simple_tokenize(text)
        
        if len(sentences) < 3:
            return []
        
        questions = []
        question_types = ['multiple_choice', 'true_false', 'fill_blank']
        
        for i in range(min(num_questions, len(sentences))):
            sentence = sentences[i % len(sentences)]
            q_type = question_types[i % len(question_types)]
            
            if q_type == 'multiple_choice':
                question = self.create_multiple_choice(sentence, i + 1)
            elif q_type == 'true_false':
                question = self.create_true_false(sentence, i + 1)
            elif q_type == 'fill_blank':
                question = self.create_fill_blank(sentence, i + 1)
            
            if question:
                questions.append(question)
        
        return questions
    
    def create_multiple_choice(self, sentence, q_id):
        """Create multiple choice question"""
        words = sentence.split()
        if len(words) < 5:
            return None
        
        # Find key word to replace
        key_word = None
        for word in words:
            if word.lower() in self.key_terms[:5] and len(word) > 3:
                key_word = word
                break
        
        if not key_word:
            key_word = words[len(words)//2]
        
        question_text = sentence.replace(key_word, "______")
        
        # Simple distractors
        distractors = [
            f"Alternative {key_word}",
            f"Modified {key_word}",
            "None of the above"
        ]
        
        options = [key_word] + distractors
        random.shuffle(options)
        correct_index = options.index(key_word)
        
        return {
            'id': q_id,
            'type': 'multiple_choice',
            'question': f"Complete the statement: {question_text}",
            'options': {chr(65 + i): opt for i, opt in enumerate(options)},
            'correct_answer': chr(65 + correct_index),
            'points': 10
        }
    
    def create_true_false(self, sentence, q_id):
        """Create true/false question"""
        is_true = random.choice([True, False])
        modified_sentence = sentence
        
        if not is_true:
            # Add negation
            if ' is ' in sentence:
                modified_sentence = sentence.replace(' is ', ' is not ', 1)
            elif ' are ' in sentence:
                modified_sentence = sentence.replace(' are ', ' are not ', 1)
            else:
                words = sentence.split()
                words.insert(len(words)//2, 'not')
                modified_sentence = ' '.join(words)
        
        return {
            'id': q_id,
            'type': 'true_false',
            'question': f"True or False: {modified_sentence}",
            'correct_answer': is_true,
            'points': 5
        }
    
    def create_fill_blank(self, sentence, q_id):
        """Create fill-in-the-blank question"""
        words = sentence.split()
        if len(words) < 4:
            return None
        
        # Replace middle word
        blank_index = len(words) // 2
        correct_answer = words[blank_index]
        words[blank_index] = "______"
        question_text = ' '.join(words)
        
        return {
            'id': q_id,
            'type': 'fill_blank',
            'question': question_text,
            'correct_answer': correct_answer,
            'points': 8
        }

def test_telecom_pdf():
    """Test with your working Telecom PDF"""
    print("🧠 TELECOM PDF QUESTION GENERATOR")
    print("=" * 50)
    
    # Use the working PDF
    pdf_path = "../Paper_project_Telecom_II.pdf"
    
    print("📖 Reading Paper_project_Telecom_II.pdf...")
    
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            
            # Extract from first 3 pages
            for i in range(min(3, len(pdf_reader.pages))):
                page_text = pdf_reader.pages[i].extract_text()
                text += page_text + "\n"
                print(f"✅ Extracted page {i+1}")
        
        print(f"✅ Total extracted: {len(text.split())} words")
        print(f"📝 Preview: {text[:200]}...")
        
        # Generate questions
        print("\n🧠 Generating Telecom questions...")
        generator = SimplifiedQuestionGenerator()
        analysis = generator.analyze_text(text)
        questions = generator.generate_questions(text, 5)
        
        # Show results
        print(f"\n📊 ANALYSIS:")
        print(f"🏷️ Domain: {analysis['domain']}")
        print(f"📈 Words: {analysis['word_count']}")
        print(f"🔑 Key terms: {', '.join(analysis['key_terms'][:5])}")
        
        print(f"\n🎉 GENERATED TELECOM QUESTIONS:")
        print("=" * 60)
        
        for q in questions:
            print(f"\n📝 QUESTION {q['id']} ({q['type'].upper()}) - {q['points']} points")
            print(f"Q: {q['question']}")
            
            if q['type'] == 'multiple_choice':
                for option, text in q['options'].items():
                    marker = "✅" if option == q['correct_answer'] else "  "
                    print(f"   {option}) {text} {marker}")
            elif q['type'] == 'true_false':
                print(f"   ✅ Answer: {q['correct_answer']}")
            elif q['type'] == 'fill_blank':
                print(f"   ✅ Answer: {q['correct_answer']}")
        
        print(f"\n🎯 SUCCESS! Generated {len(questions)} questions from your Telecom PDF!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_telecom_pdf()