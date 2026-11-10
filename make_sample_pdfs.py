"""Creates the two sample PDFs in pdfs/ (text written by me). Needs: pip install fpdf2"""

from fpdf import FPDF

DOCS = {
    "leave_policy.pdf": [
        "Leave policy. Every full-time employee gets 24 days of paid leave a year. Leave is "
        "added at 2 days per month. Up to 10 unused days can be carried into the next year.",
        "Sick leave. Employees get 8 days of sick leave a year. A doctor's certificate is "
        "needed for sick leave longer than 2 days in a row. Sick leave cannot be carried over.",
        "Public holidays. The office is closed on 12 public holidays, including Republic Day, "
        "Independence Day, Gandhi Jayanti, Diwali and Sankranti.",
    ],
    "travel_policy.pdf": [
        "Travel policy. Employees may travel by train in 3AC or by economy flight when the "
        "train journey is longer than 12 hours. All travel must be approved by a manager first.",
        "Daily allowance. The daily allowance is Rs 1,500 in metro cities and Rs 1,000 in other "
        "cities. Hotel bills are paid up to Rs 4,000 per night with a receipt.",
    ],
}

for name, pages in DOCS.items():
    pdf = FPDF()
    pdf.set_font("Helvetica", size=12)
    pdf.set_auto_page_break(False)  # keep the footer on the same page
    for number, text in enumerate(pages, start=1):
        pdf.add_page()
        pdf.cell(0, 8, "GenAI4U Sample Co. - HR policies")  # a header on every page
        pdf.ln(12)
        pdf.multi_cell(0, 8, text)
        pdf.set_y(-20)
        pdf.cell(0, 8, "Internal use only")  # a footer on every page
    pdf.output(f"pdfs/{name}")
    print("wrote", name)
