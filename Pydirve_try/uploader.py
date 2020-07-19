from pydrive.auth import GoogleAuth

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


from pydrive.drive import GoogleDrive

# upload one file and get download link
def uploadFileGetUrl(gauth, path, name):
    drive = GoogleDrive(gauth)
    file1 = drive.CreateFile({'title': name})  # Create GoogleDriveFile instance with title file.
 
    file1.SetContentFile(path)
    file1.Upload()

    file_list = drive.ListFile({'q': "'root' in parents"}).GetList()
    
    return file_list[0]["webContentLink"] # return the download link


# upload all files in a directory and get a list of download links:
from os import listdir
from os.path import isfile, join

def uploadFiles(gauth, directory):
    # drive = GoogleDrive(gauth)
    files = [f for f in listdir(directory) if isfile(join(directory,f))]
    urls = []
    for file in files:
        urls.append((file,uploadFileGetUrl(gauth,join(directory,file),file)))
    return urls # return a list of tuples [(filename, download link),]


# sample usage
directory = "./pdf_test_data"
print(uploadFiles(gauth,directory))




