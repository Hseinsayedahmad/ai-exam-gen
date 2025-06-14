"""
Choose Any PDF Test - Interactive PDF Selection
"""

import PyPDF2
import os
from working_pdf_test import SimplifiedQuestionGenerator

def choose_and_test_pdf():
    print("🧠 INTERACTIVE PDF QUESTION GENERATOR")
    print("=" * 50)
    
    # Find all PDFs
    main_folder = "C:\\Users\\Lenovo\\Desktop\\New folder\\AI-Exam-Generator\\backend\\pdf"
    pdf_files = [f for f in os.listdir(main_folder) if f.endswith('.pdf')]
    
    if not pdf_files:
        print("❌ No PDFs found in project folder!")
        return
    
    print(f"📁 Found {len(pdf_files)} PDF files:")
    for i, pdf in enumerate(pdf_files):
        print(f"   {i+1}. {pdf}")
    
    # Let user choose
    try:
        print(main_folder)
        choice = int(input(f"\n🎯 Choose PDF (1-{len(pdf_files)}): ")) - 1
        if 0 <= choice < len(pdf_files):
            selected_pdf = pdf_files[choice]
        else:
            print("❌ Invalid choice, using first PDF")
            selected_pdf = pdf_files[0]
    except:
        print("❌ Invalid input, using first PDF")
        selected_pdf = pdf_files[0]
    
    print(f"\n📄 Selected: {selected_pdf}")
    
    # Ask for number of questions
    try:
        num_q = int(input("🎯 How many questions? (1-10): "))
        num_q = max(1, min(10, num_q))
    except:
        num_q = 5
    
    # Process selected PDF
    pdf_path = os.path.join(main_folder, selected_pdf)
    
    print(f"\n📖 Reading {selected_pdf}...")
    
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            
            # Extract text
            max_pages = min(3, len(pdf_reader.pages))
            for i in range(max_pages):
                page_text = pdf_reader.pages[i].extract_text()
                if page_text and len(page_text.strip()) > 10:
                    text += page_text + "\n"
                    print(f"✅ Page {i+1}: {len(page_text)} chars")
                else:
                    print(f"⚠️ Page {i+1}: No readable text")
        
        if len(text.split()) < 20:
            print("❌ Not enough readable text extracted!")
            return
        
        print(f"✅ Extracted {len(text.split())} words total")
        
        # Generate questions
        print(f"\n🧠 Generating {num_q} questions...")
        generator = SimplifiedQuestionGenerator()
        analysis = generator.analyze_text(text)
        questions = generator.generate_questions(text, num_q)
        
        # Show results
        print(f"\n📊 ANALYSIS OF {selected_pdf.upper()}:")
        print(f"🏷️ Domain: {analysis['domain']}")
        print(f"📈 Words: {analysis['word_count']}")
        print(f"🔑 Key terms: {', '.join(analysis['key_terms'][:5])}")
        
        print(f"\n🎉 QUESTIONS FROM {selected_pdf.upper()}:")
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
        
        print(f"\n🎯 SUCCESS! Generated {len(questions)} questions!")
        
        # Ask if user wants to test another PDF
        again = input("\n🔄 Test another PDF? (y/n): ").lower()
        if again == 'y':
            choose_and_test_pdf()
        
    except Exception as e:
        print(f"❌ Error processing {selected_pdf}: {e}")

if __name__ == "__main__":
    choose_and_test_pdf()