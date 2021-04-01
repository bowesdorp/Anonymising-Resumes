import textract





if __name__ == '__main__':
    path = '/Users/bowesdorp/PycharmProjects/Anonymising-Resumes/data/psuedonised/docx/105.docx'
    text = textract.process(path)
    print(text)

    f = open("./test2.txt", "w+")
    f.write(str(text))
    f.close()
