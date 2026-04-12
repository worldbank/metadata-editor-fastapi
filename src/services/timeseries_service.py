class TimeseriesService:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def get_db_path(self) -> str:
        return self.db_path
