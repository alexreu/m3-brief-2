from pydantic import BaseModel, Field
from datetime import date
from decimal import Decimal
from typing import Annotated

from .loan_information import LoanInformationRead, LoanInformationBase

FixedDecimal = Annotated[Decimal, Field(decimal_places=2)]


class ClientBase(BaseModel):
    first_name: str = Field(examples=["John"])
    last_name: str = Field(examples=["Doe"])
    date_of_birth: date
    height_cm: FixedDecimal = Field(examples=["175.5"])
    weight_kg: FixedDecimal = Field(examples=["70.5"])
    sex: str = Field(examples=["H", "F"])
    has_sport_license: bool
    education_level: str = Field(
        examples=["aucun", "bac", "bac+2", "bac+3", "master", "doctorat"])
    region: str = Field(examples=["Île-de-France", "Pays de la Loire"])
    is_smoker: bool
    is_french_national: bool
    family_status: str = Field(
        examples=["célibataire", "marié", "divorcé", "veuf"])


class ClientCreate(ClientBase):
    pass


class ClientCreateWithLoanInformations(ClientBase):
    loan_informations: list[LoanInformationBase] = Field(examples=[[
        {
            "estimated_monthly_income": "3000.00",
            "credit_history_count": 5,
            "personal_risk_score": "0.5",
            "credit_score": 750,
            "monthly_rent": "1000.00",
            "loan_amount": "850.90",
            "account_created_at": "2020-01-01"
        }]])


class ClientRead(ClientBase):
    id: int
    created_at: date

    class Config:
        from_attributes = True


class ClientWithLoanInformations(ClientRead):
    loan_informations: list[LoanInformationRead] = []
