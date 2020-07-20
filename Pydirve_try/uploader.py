from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

from os import listdir
from os.path import isfile, join, basename, split, abspath

from pdf2image import convert_from_path, convert_from_bytes
import pytesseract
from PIL import Image
import json

gauth = GoogleAuth()

# Try to load saved client credentials
gauth.LoadCredentialsFile("mycreds.txt")
if gauth.credentials is None:
    # Authenticate if they're not there
    gauth.LocalWebserverAuth()
elif gauth.access_token_expired:
    # Refresh them if expired
    gauth.Refresh()
else:
    # Initialize the saved creds
    gauth.Authorize()
# Save the current credentials to a file
gauth.SaveCredentialsFile("mycreds.txt")



# upload one file and get download link
def uploadFileGetUrl(gauth, path, name):
    drive = GoogleDrive(gauth)
    file1 = drive.CreateFile({'title': name})  # Create GoogleDriveFile instance with title file.
 
    file1.SetContentFile(path)
    file1.Upload()

    file_list = drive.ListFile({'q': "'root' in parents"}).GetList()
    
    return file_list[0]["webContentLink"] # return the download link


# upload all files in a directory and get a list of download links:
def uploadFiles(gauth, directory):
    # drive = GoogleDrive(gauth)
    files = [f for f in listdir(directory) if isfile(join(directory,f))]
    urls = []
    for file in files:
        urls.append((file,uploadFileGetUrl(gauth,join(directory,file),file)))
    return urls # return a list of tuples [(filename, download link),]


# extract text from one xxx.pdf and save it as xxx.txt at the same folder, return name of the text file
def extractTextFromPdf(path):
    img_list = convert_from_path(path)

    whole_text = ""
    print(f'start converting {path} ...')
    count = 1
    for img in img_list:
        # img.show()
        str_gen = pytesseract.image_to_string(img)
        whole_text += str_gen
        print("finish page", count)
        count+=1
    
    filename = basename(path)[:-4] + '.txt'
    print("write to",filename)

    with open(filename,'w+') as f:
        json.dump(whole_text,f)
    
    # return abspath(filename)
    return filename

def extractTextFromDirectory(directory):
    files = [f for f in listdir(directory) if isfile(join(directory,f))]
    file_list = []
    for f in files:
        file_list.append(extractTextFromPdf(join(directory,f)))
    return file_list

def createLog(urls,extracted_files):
    with open('Log.txt','w+') as f:
        pairs = {}
        for i in range(len(urls)):
            pairs[urls[i][0]] = (urls[i][1],extracted_files[i])
        json.dump(pairs,f)
# sample usage
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

# directory = "./pdf_test_data"
# print(uploadFiles(gauth,directory))

# print(extractTextFromPdf('./pdf_test_data/multipage_small.pdf'))
# print(extractTextFromDirectory('./pdf_test_data'))

# urls = [(i,"hello") for i in range(5)]
# extracted_text = extractTextFromDirectory('./pdf_test_data')
# createLog(urls,extracted_text)




