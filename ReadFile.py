import textract
import os





if __name__ == '__main__':
    path_1 = '/Users/bowesdorp/PycharmProjects/Anonymising-Resumes/data/psuedonised/docx/'
    path_2 = '/Users/bowesdorp/PycharmProjects/Anonymising-Resumes/data/psuedonised2/'

    for (dirpath, dirnames, filenames) in os.walk(path_1):
        for filename in filenames:
            split = filename.split('.')
            if split[1] == 'docx':
                print(filename)
                text = textract.process(path_1+filename)
                t = text.decode('utf-8').split()

                new_string = " ".join(t)
                print(new_string)

                f = open(path_2 + split[0] + ".txt", "w+")
                f.write(new_string)
                f.close()

            # print(text)
            # t = text.decode('utf-8')
            #
            # f = open(path_2 + split[0] + ".txt", "w+")
            # f.write(t)
            # f.close()
