#!/usr/bin/env python3
"""
ResQAI — Sample PDF Generator
Creates sample disaster management PDFs for testing the RAG pipeline.
Run this script to generate test PDFs before running the app.

Usage: python create_sample_pdfs.py
"""

import os

def create_sample_pdfs():
    """Generate sample disaster management PDFs using reportlab or fpdf."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        use_reportlab = True
    except ImportError:
        try:
            from fpdf import FPDF
            use_reportlab = False
        except ImportError:
            print("Neither reportlab nor fpdf is installed.")
            print("Creating plain text files instead...")
            create_txt_files()
            return

    sample_dir = "sample_docs"
    os.makedirs(sample_dir, exist_ok=True)

    # Read the main guide
    guide_path = os.path.join(sample_dir, "disaster_management_guide.txt")
    if not os.path.exists(guide_path):
        print(f"Guide file not found at {guide_path}")
        return

    with open(guide_path, "r") as f:
        content = f.read()

    # Split into sections
    sections = content.split("\nCHAPTER ")
    intro = sections[0]
    chapters = ["CHAPTER " + s for s in sections[1:]]

    if use_reportlab:
        _create_with_reportlab(sample_dir, intro, chapters)
    else:
        _create_with_fpdf(sample_dir, intro, chapters)

    print("✅ Sample PDFs created in sample_docs/ folder")
    print("   Upload these to ResQAI to test the RAG pipeline")


def _create_with_reportlab(sample_dir, intro, chapters):
    """Create PDFs using ReportLab."""
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch

    styles = getSampleStyleSheet()

    for i, chapter in enumerate(chapters[:3], 1):
        lines = chapter.strip().split('\n')
        title = lines[0].strip()
        filename = os.path.join(sample_dir, f"chapter_{i}_{title[:20].replace(' ', '_').lower()}.pdf")

        doc = SimpleDocTemplate(filename, pagesize=letter)
        story = []

        story.append(Paragraph("ResQAI Sample Document", styles['Title']))
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph(title, styles['Heading1']))
        story.append(Spacer(1, 0.1*inch))

        for line in lines[1:]:
            if line.strip():
                if line.isupper() and len(line) > 10:
                    story.append(Paragraph(line, styles['Heading2']))
                else:
                    story.append(Paragraph(line, styles['Normal']))
                story.append(Spacer(1, 0.05*inch))

        doc.build(story)
        print(f"   Created: {filename}")


def _create_with_fpdf(sample_dir, intro, chapters):
    """Create PDFs using FPDF."""
    from fpdf import FPDF

    for i, chapter in enumerate(chapters[:3], 1):
        lines = chapter.strip().split('\n')
        title = lines[0].strip()
        filename = os.path.join(sample_dir, f"chapter_{i}_{title[:20].replace(' ', '_').lower()}.pdf")

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "ResQAI Sample Document", ln=True, align='C')
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, title[:60], ln=True)
        pdf.set_font("Arial", "", 11)

        for line in lines[1:]:
            if line.strip():
                # Encode safely for Latin-1
                safe_line = line.encode('latin-1', errors='replace').decode('latin-1')
                pdf.multi_cell(0, 7, safe_line)

        pdf.output(filename)
        print(f"   Created: {filename}")


def create_txt_files():
    """Fallback: create plain text files if PDF libraries are unavailable."""
    sample_dir = "sample_docs"
    os.makedirs(sample_dir, exist_ok=True)
    print("Plain text files already exist in sample_docs/")
    print("Install reportlab to create PDFs: pip install reportlab")


if __name__ == "__main__":
    create_sample_pdfs()
