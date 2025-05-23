# Employee Onboarding Document Verification System

A secure and efficient system for onboarding employees by verifying their documents using blockchain technology and AI-powered information extraction. This system ensures the authenticity of employee-submitted documents while maintaining immutable records on the blockchain.

## Project Overview

This system enables:
- Secure document upload and verification for employee onboarding.
- AI-powered extraction of key employee details from documents (e.g., name, ID number, address, etc.).
- Immutable record-keeping of verified documents on a blockchain network.
- A web interface for HR teams to manage employee documents efficiently.
- Comprehensive tracking and logging of all onboarding activities.


## Features

- Document Verification
- Blockchain Integration
- AI-Powered Information Extraction
- User-Friendly Web Interface
- Secure file storage
- Error Handling & Logging

## Project Structure

```
EODVS/
├── app.py                  # Main Flask application
├── config.py              # Configuration settings
├── requirements.txt       # Project dependencies
├── .env                   # Environment variables
├── static/               # Static files
├── templates/            # HTML templates
├── uploads/              # Document upload directory
└── document_verification.db  # SQLite database
```

## Technologies Used
- Python 3.x
- Flask (Web Framework)
- Web3 (Blockchain Integration)
- Google Generative AI
- SQLite (Database)
- HTML/CSS/JavaScript

## Setup and Installation

1. Clone the repository
2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables in `.env`:
- GOOGLE_API_KEY
- Other necessary blockchain configuration

5. Initialize the database and start the application:
```bash
python app.py
```

## Usage

- Open the web application at http://localhost:5000.
- For HR Teams:
    - Upload employee onboarding documents (e.g., IDs, contracts).
    - Verify extracted information and approve documents.
- Verification Results:
    - View blockchain transaction details for each verified document.
    - Check the history of uploaded documents and verification activities.

## Additional Enhancements

- Email Notifications: Notify employees about their onboarding status via email.
- Admin Dashboard: A dashboard to view onboarding metrics and track progress.
- Integration with HR Systems: Sync with existing HR tools or databases for seamless onboarding.


<!-- —---------
The system demonstrated 94.1% average accuracy in extracting key information such as name, date of birth, document type and ID from any uploaded government documents using Google’s Gemini API. This approach outperformed traditional OCR methods, although minor errors occurred with handwritten or low-resolution documents, indicating strong but improvable AI performance.
—------------ -->
