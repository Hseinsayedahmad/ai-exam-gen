"""
PDF Test for Rule-Based Model
Test your advanced rule-based model with real PDFs!
"""

import PyPDF2
import io
from rule_based_model import AdvancedRuleBasedQuestionGenerator

def extract_pdf_text(pdf_path):
    """Extract text from PDF file"""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            
            print(f"📄 PDF has {len(pdf_reader.pages)} pages")
            
            # Extract text from all pages
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                text += page_text + "\n"
                print(f"✅ Extracted page {page_num + 1}")
            
            return text.strip()
    
    except Exception as e:
        print(f"❌ Error reading PDF: {e}")
        return None

def test_with_pdf():
    """Test the rule-based model with a real PDF"""
    
    print("🧠 PDF RULE-BASED MODEL TESTER")
    print("=" * 50)
    
    # Ask for PDF path
    pdf_path = input("📁 Enter PDF file path (or drag PDF here): ").strip().strip('"')
    
    if not pdf_path.endswith('.pdf'):
        print("❌ Please provide a PDF file!")
        return
    
    # Extract text
    print("\n📖 Extracting text from PDF...")
    text = extract_pdf_text(pdf_path)
    
    if not text:
        print("❌ Could not extract text from PDF")
        return
    
    print(f"✅ Extracted {len(text.split())} words")
    print(f"📝 Preview: {text[:200]}...")
    
    # Initialize model
    print("\n🧠 Initializing Advanced Rule-Based Model...")
    generator = AdvancedRuleBasedQuestionGenerator()
    
    # Ask for number of questions
    try:
        num_q = int(input("\n🎯 How many questions to generate? (1-20): "))
        num_q = max(1, min(20, num_q))
    except:
        num_q = 5
    
    # Ask for difficulty
    difficulty = input("⚡ Difficulty (easy/medium/hard): ").lower()
    if difficulty not in ['easy', 'medium', 'hard']:
        difficulty = 'medium'
    
    # Generate questions
    print(f"\n🚀 Generating {num_q} {difficulty} questions...")
    questions = generator.generate_questions(text, num_q, difficulty)
    
    # Display results
    print("\n" + "=" * 60)
    print("🎉 GENERATED QUESTIONS FROM YOUR PDF:")
    print("=" * 60)
    
    for i, q in enumerate(questions, 1):
        print(f"\n📝 QUESTION {i} ({q['type'].upper()}) - {q['points']} points")
        print(f"Q: {q['question']}")
        
        if q['type'] == 'multiple_choice':
            for option, text in q['options'].items():
                marker = "✅" if option == q['correct_answer'] else "  "
                print(f"   {option}) {text} {marker}")
        
        elif q['type'] == 'true_false':
            print(f"   ✅ Answer: {q['correct_answer']}")
        
        elif q['type'] in ['fill_blank', 'definition']:
            print(f"   ✅ Answer: {q['correct_answer']}")
        
        if 'quality_score' in q:
            print(f"   📊 Quality: {q['quality_score']:.1f}/100")
    
    # Show statistics
    stats = generator.get_generation_statistics()
    print(f"\n📊 ANALYSIS RESULTS:")
    print(f"🏷️  Domain: {stats['domain_classification']}")
    print(f"🔑 Key Terms: {', '.join(generator.key_terms[:5])}")
    print(f"📈 Document Complexity: {stats['document_analysis'].get('complexity_score', 0):.1f}")
    
    print(f"\n🎯 SUCCESS! Generated {len(questions)} questions from your PDF!")

if __name__ == "__main__":
    test_with_pdf()