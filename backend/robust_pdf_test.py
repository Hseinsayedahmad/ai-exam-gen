"""
Robust PDF Test with Better Error Handling
"""

import PyPDF2
import os

def extract_pdf_text_safe(pdf_path):
    """Safely extract text from PDF"""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            
            print(f"📄 PDF has {len(pdf_reader.pages)} pages")
            
            # Try all pages until we get text
            for i in range(min(5, len(pdf_reader.pages))):
                try:
                    page_text = pdf_reader.pages[i].extract_text()
                    if page_text and len(page_text.strip()) > 10:
                        text += page_text + "\n"
                        print(f"✅ Page {i+1}: {len(page_text)} characters")
                    else:
                        print(f"⚠️ Page {i+1}: No readable text (might be image)")
                except:
                    print(f"❌ Page {i+1}: Error extracting")
            
            return text.strip()
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_pdfs():
    print("🧠 TESTING PDF TEXT EXTRACTION")
    print("=" * 50)
    
    # Find PDFs
    main_folder = ".."
    pdf_files = [f for f in os.listdir(main_folder) if f.endswith('.pdf')]
    
    print(f"📁 Found {len(pdf_files)} PDFs:")
    for i, pdf in enumerate(pdf_files):
        print(f"   {i+1}. {pdf}")
    
    # Test each PDF
    for pdf_file in pdf_files:
        print(f"\n🔍 TESTING: {pdf_file}")
        print("-" * 40)
        
        pdf_path = os.path.join(main_folder, pdf_file)
        text = extract_pdf_text_safe(pdf_path)
        
        if text and len(text.split()) > 20:
            print(f"✅ SUCCESS! Extracted {len(text.split())} words")
            print(f"📝 Preview: {text[:200]}...")
            
            # Test with simple question generation
            print("\n🧠 Testing simple question generation...")
            sentences = text.split('.')[:3]  # First 3 sentences
            for i, sent in enumerate(sentences):
                if len(sent.split()) > 5:
                    words = sent.split()
                    middle_word = words[len(words)//2]
                    question = sent.replace(middle_word, "______")
                    print(f"Q{i+1}: {question}?")
                    print(f"A{i+1}: {middle_word}")
            
            print(f"\n🎉 {pdf_file} works perfectly!")
            break
        else:
            print(f"❌ Could not extract readable text from {pdf_file}")
    
    print("\n✅ PDF testing complete!")

if __name__ == "__main__":
    test_pdfs()