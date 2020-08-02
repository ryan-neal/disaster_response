from dotenv import load_dotenv
load_dotenv()
import os

def get_database_url():
    user = os.environ.get("user")
    password = os.environ.get("pass")
    server = os.environ.get("server")
    url = 'mysql://'+ user + ':' +password+ '@' + server + '/categories'
    return url