from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from config.db import get_db
from models.loan_information import LoanInformation
from schemas.loan_information import LoanInformationRead

router = APIRouter(prefix="/loan_informations", tags=["loan_informations"])


@router.get("/", response_model=list[LoanInformationRead], summary="List all loan informations")
def get_loan_informations(db: Session = Depends(get_db)):

    loan_informations = db.query(LoanInformation).all()

    return loan_informations


@router.get("/{loan_information_id}", response_model=LoanInformationRead, summary="Get loan information details")
def get_loan_information(loan_information_id: int, db: Session = Depends(get_db)):
    loan_information = db.query(LoanInformation).filter(
        LoanInformation.id == loan_information_id).first()

    if not loan_information:
        raise HTTPException(
            status_code=404, detail="Loan information not found")

    return loan_information


@router.delete("/{loan_information_id}", status_code=204, summary="Delete a loan information")
def delete_loan_information(loan_information_id: int, db: Session = Depends(get_db)):
    loan_information = db.query(LoanInformation).filter(
        LoanInformation.id == loan_information_id).first()

    if not loan_information:
        raise HTTPException(
            status_code=404, detail="Loan information not found")

    db.delete(loan_information)
    db.commit()
