from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session, selectinload
from config.db import get_db
from models.client import Client
from schemas.client import ClientRead, ClientWithLoanInformations, ClientCreateWithLoanInformations
from models.loan_information import LoanInformation
from datetime import date
from sqlalchemy.exc import SQLAlchemyError

router = APIRouter(prefix="/clients", tags=["clients"])


@router.get("/", response_model=list[ClientRead], status_code=200, summary="List all clients")
def get_clients(db: Session = Depends(get_db)):

    clients = db.query(Client).all()

    return clients


@router.get("/{client_id}", response_model=ClientWithLoanInformations, status_code=200, summary="Get client details with loan information")
def get_client(client_id: int, db: Session = Depends(get_db)):
    client = db.query(Client).filter(Client.id == client_id).first()

    if not client:
        raise HTTPException(status_code=404, detail="Client not found")

    return client


@router.post("/", response_model=ClientWithLoanInformations, status_code=201, summary="Create a new client")
def create_client(payload: ClientCreateWithLoanInformations, db: Session = Depends(get_db)):

    today = date.today()
    age = today.year - payload.date_of_birth.year - (
        (today.month, today.day) < (
            payload.date_of_birth.month, payload.date_of_birth.day)
    )

    if age < 18:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Client must be at least 18 years old")

    client_data = payload.model_dump(exclude={"loan_informations"})
    client_data["age"] = age
    client_data["created_at"] = today

    db_client = Client(**client_data)

    try:
        db.add(db_client)
        db.flush()  # Flush to get the generated ID for the client

        for loan_info in payload.loan_informations:
            loan_info_data = loan_info.model_dump()
            loan_info_data["client_id"] = db_client.id
            loan_info_data["created_at"] = today
            db.add(LoanInformation(**loan_info_data))

        db.commit()

        return db.query(Client).options(selectinload(Client.loan_informations)).filter(Client.id == db_client.id).first()
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database error: " + str(e))


@router.delete("/{client_id}", status_code=204, summary="Delete a client")
def delete_client(client_id: int, db: Session = Depends(get_db)):
    client = db.query(Client).filter(Client.id == client_id).first()

    if not client:
        raise HTTPException(status_code=404, detail="Client not found")

    try:
        db.delete(client)
        db.commit()
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database error: " + str(e))
