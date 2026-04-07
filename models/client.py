from .base import Base

from sqlalchemy import Column, Integer, String, Enum, Boolean, Numeric, Date
from sqlalchemy.orm import relationship


class Client(Base):
    __tablename__ = 'clients'

    id = Column(Integer, primary_key=True)
    first_name = Column(String)
    last_name = Column(String)
    age = Column(Integer)
    date_of_birth = Column(Date)
    height_cm = Column(Numeric)
    weight_kg = Column(Numeric)
    sex = Column(Enum('H', 'F', name="sex_enum"))
    has_sport_license = Column(Boolean)
    education_level = Column(
        Enum('bac', 'bac+2', 'bac+3', 'master', 'doctorat', 'aucun', name="education_level_enum"))
    region = Column(String)
    is_smoker = Column(Boolean)
    is_french_national = Column(Boolean)
    family_status = Column(
        Enum('marié', 'célibataire', 'divorcé', 'veuf', name="family_status_enum"))
    created_at = Column(Date)

    loan_informations = relationship(
        "LoanInformation", back_populates="client")

    def __repr__(self):
        return f"<Client(id={self.id}, first_name='{self.first_name}', last_name='{self.last_name}')>"
