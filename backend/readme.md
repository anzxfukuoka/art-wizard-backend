## Development setup

Windows
```bash
pip install -r requirements.txt
venv\Scripts\activate
python setup.py
```

Docker

```bash
docker build -t gi-tool-backend .
docker run -p 5000:5000 --env-file .env gi-tool-backend
```