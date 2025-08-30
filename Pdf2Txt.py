import os
import PyPDF2

# 设置PDF文件所在目录和输出TXT文件的目录
pdf_dir = './data_pdf'
txt_dir = './data_txt'

# 创建输出目录（如果不存在）
os.makedirs(txt_dir, exist_ok=True)

# 遍历目录中的所有PDF文件
for filename in os.listdir(pdf_dir):
    if filename.endswith('.pdf'):
        pdf_path = os.path.join(pdf_dir, filename)
        txt_path = os.path.join(txt_dir, filename.replace('.pdf', '.txt'))

        with open(pdf_path, 'rb') as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            with open(txt_path, 'w', encoding='utf-8') as txt_file:
                for page in reader.pages:
                    txt_file.write(page.extract_text() + '\n')


