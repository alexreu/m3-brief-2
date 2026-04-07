from fastapi import FastAPI
from api.routes.clients import router as clients_router
from api.routes.loan_informations import router as loan_informations_router

app = FastAPI(title="Loan Management API",
              description="API for managing clients and their loans", version="1.0.0")

app.include_router(clients_router)
app.include_router(loan_informations_router)
