import os
import shutil


def sort_files():
    print(f'Starting to sort files...')
    path = os.getcwd() + '/data/raw/Copy_resumes'
    print(f'Path to data: {path}.')
    path_pdf = os.getcwd() + '/data/pdf'
    path_doc = os.getcwd() + '/data/doc'
    path_docx = os.getcwd() + '/data/docx'
    print(f'Creating folders...')

    create_folder(path_pdf)
    create_folder(path_doc)
    create_folder(path_docx)
    counter = 1
    for root, dirs, files in os.walk(path):
        for filename in files:
            split = filename.split('.')
            doc_type = split[len(split)-1]
            filepath = path + '/' + filename
            if doc_type == 'pdf':
                # print(path_pdf + '/' + str(counter) + '.pdf')
                shutil.copy2(filepath, path_pdf + '/' + str(counter) + '.pdf')
            elif doc_type == 'doc':
                # print(path_doc + '/' + str(counter) + '.doc')
                shutil.copy2(filepath, path_doc + '/' + str(counter) + '.doc')
            elif doc_type == 'docx':
                # print(path_docx + '/' + str(counter) + '.docx')
                shutil.copy2(filepath, path_docx + '/' + str(counter) + '.docx')
            counter += 1
    return 1


def create_folder(path):
    try:
        os.mkdir(path)
    except OSError:
        shutil.rmtree(path)
        os.mkdir(path)
        print("Successfully created the directory %s " % path)
    else:
        print("Successfully created the directory %s " % path)
    return


def convert_file(path):
    return 1


if __name__ == '__main__':
    sort_files()
