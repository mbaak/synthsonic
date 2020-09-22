from typing import List

from api.synthesis import get_data
from fastapi import FastAPI, File
from starlette.responses import Response

app = FastAPI(title="Synthsonic - synthetic data generator service",
              description='''Obtain synthetic, safely shareable and indistinguishable-from-real synthetic data.''',
              version="0.1.0",
              )


@app.post("/synthesize")
def get_synthetic_data(file: bytes = File(...), rows: List[int] = [0], categorical_columns: List[str] = [],
                       ordinal_columns: List[str] = [],
                       ):
    _rows = rows[0]

    _categorical_columns = categorical_columns[0].split(',')
    _ordinal_columns = ordinal_columns[0].split(',')

    synthetic_data = get_data(file, _categorical_columns, _ordinal_columns, _rows + 1)

    return Response(synthetic_data, media_type="text/csv")
