from sqlalchemy.orm import Session

from app.db.schema import User

class UserService:
    def __init__(self, session: Session):
        self.db = session
    
    def list_users(self) -> list[User]:
        return self.db.query(User).all()

    def get_user(self, user_id: int) -> User:
        return self.db.query(User).filter(User.id == user_id).first()
