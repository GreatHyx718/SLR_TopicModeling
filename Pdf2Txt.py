import os
import PyPDF2


def pdf_to_txt(pdf_dir, txt_dir):
    for filename in os.listdir(pdf_dir):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(pdf_dir, filename)
            txt_path = os.path.join(txt_dir, filename.replace('.pdf', '.txt'))

            with open(pdf_path, 'rb') as pdf_file:
                reader = PyPDF2.PdfReader(pdf_file)
                with open(txt_path, 'w', encoding='utf-8') as txt_file:
                    for page in reader.pages:
                        txt_file.write(page.extract_text() + '\n')




if __name__ == '__main__':
    pdfDir = './data_pdf'
    txtDir = './data_txt'
    os.makedirs(txtDir, exist_ok=True) # create the output dir if it does not exist
    pdf_to_txt(pdfDir, txtDir)

