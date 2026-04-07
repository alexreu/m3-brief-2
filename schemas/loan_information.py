from pydantic import BaseModel, Field
from datetime import date
from decimal import Decimal
from typing import Annotated

FixedDecimal = Annotated[Decimal, Field(decimal_places=2)]


class LoanInformationBase(BaseModel):
    estimated_monthly_income: FixedDecimal = Field(examples=["3000.00"])
    credit_history_count: int = Field(examples=[5])
    personal_risk_score: Decimal = Field(examples=["0.5"])
    credit_score: int = Field(examples=[750])
    monthly_rent: FixedDecimal = Field(examples=["1000.00"])
    loan_amount: FixedDecimal = Field(examples=["850.90"])
    account_created_at: date


class LoanInformationCreate(LoanInformationBase):
    client_id: int


class LoanInformationRead(LoanInformationBase):
    id: int
    client_id: int
    created_at: date

    class Config:
        from_attributes = True
