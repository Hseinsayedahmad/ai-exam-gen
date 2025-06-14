"""
Simple PDF Test - Just put PDF in main folder!
"""

import PyPDF2
from rule_based_model import AdvancedRuleBasedQuestionGenerator
import os

def test_with_pdf():
    print("🧠 PDF RULE-BASED MODEL TESTER")
    print("=" * 50)
    
    # Look for PDFs in main project folder
    main_folder = ".."  # Go up one level from backend
    pdf_files = [f for f in os.listdir(main_folder) if f.endswith('.pdf')]
    
    print(f"📁 Found {len(pdf_files)} PDF files:")
    for i, pdf in enumerate(pdf_files):
        print(f"   {i+1}. {pdf}")
    
    if not pdf_files:
        print("❌ No PDFs found! Put a PDF in your AI-Exam-Generator folder")
        return
    
    # Use first PDF
    selected_pdf = pdf_files[0]
    print(f"📄 Using: {selected_pdf}")
    
    pdf_path = os.path.join(main_folder, selected_pdf)
    
    # Extract text
    print(f"\n📖 Reading {selected_pdf}...")
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            
            print(f"📄 PDF has {len(pdf_reader.pages)} pages")
            
            # Extract from first 3 pages
            max_pages = min(3, len(pdf_reader.pages))
            for i in range(max_pages):
                page_text = pdf_reader.pages[i].extract_text()
                text += page_text + "\n"
                print(f"✅ Extracted page {i+1}")
                
    except Exception as e:
        print(f"❌ Error reading PDF: {e}")
        return
    
    print(f"✅ Extracted {len(text.split())} words")
    
    # Generate questions
    print("\n🧠 Generating questions...")
    generator = AdvancedRuleBasedQuestionGenerator()
    questions = generator.generate_questions(text, 5, 'medium')
    
    print(f"\n🎉 Generated {len(questions)} questions from your PDF!")

if __name__ == "__main__":
    test_with_pdf()