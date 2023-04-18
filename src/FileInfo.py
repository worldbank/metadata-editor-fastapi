from pydantic import BaseModel

class FileInfo(BaseModel):
    file_path: str