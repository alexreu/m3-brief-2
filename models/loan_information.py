from .base import Base

from sqlalchemy import Column, Integer, Date, Numeric,  ForeignKey
from sqlalchemy.orm import relationship


class LoanInformation(Base):
    __tablename__ = 'loan_informations'

    id = Column(Integer, primary_key=True)
    client_id = Column(Integer, ForeignKey('clients.id'))
    estimated_monthly_income = Column(Numeric)
    credit_history_count = Column(Integer)
    personal_risk_score = Column(Numeric)
    credit_score = Column(Integer)
    monthly_rent = Column(Numeric)
    loan_amount = Column(Numeric)
    account_created_at = Column(Date)
    created_at = Column(Date)

    client = relationship("Client", back_populates="loan_informations")

    def __repr__(self):
        return f"<LoanInformation(id={self.id}, client_id={self.client_id})>"
