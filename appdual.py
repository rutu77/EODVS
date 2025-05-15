import os
import time
import hashlib
import sqlite3
import base64
import json
import re
import logging
from web3 import Web3
from dotenv import load_dotenv
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template, flash, redirect, url_for, session, Response
from functools import wraps
from config import Config, validate_participant_name, validate_document_hash

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.urandom(24)

config = Config()
web3 = config.get_web3()


# Load Google API key
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Dummy HR credentials (In production, use secure user storage)
HR_USERNAME = 'admin'
HR_PASSWORD = 'admin123'

# Blockchain details
neoxt_url = "https://neoxt4seed1.ngd.network"
web3 = Web3(Web3.HTTPProvider(neoxt_url))
from_address = "0x8883bFFa42A7f5B509D0929c6fFa041e46E18e2f"
private_key = "9b63cd445ab8312da178e90693290d0d2c98a334f77634013f5d8cfce60f644f"
chain_id = 12227332

dictionary = {}

# DB initialization
conn = sqlite3.connect('document_verification.db')
c = conn.cursor()
c.execute('''
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        participant_name TEXT,
        document_hash TEXT,
        txn_hash TEXT,
        filename TEXT,
        timestamp TEXT
    )
''')
c.execute('''
    CREATE TABLE IF NOT EXISTS original_documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        participant_name TEXT,
        document_hash TEXT,
        document_content BLOB,
        filename TEXT,
        timestamp TEXT
    )
''')
#c.execute('ALTER TABLE original_documents ADD COLUMN filename TEXT')
#c.execute("ALTER TABLE documents ADD COLUMN filename TEXT")

try:
    c.execute("ALTER TABLE documents ADD COLUMN full_name TEXT")
except sqlite3.OperationalError:
    pass
try:
    c.execute("ALTER TABLE documents ADD COLUMN date_of_birth TEXT")
except sqlite3.OperationalError:
    pass
try:
    c.execute("ALTER TABLE documents ADD COLUMN document_type TEXT")
except sqlite3.OperationalError:
    pass
try:
    c.execute("ALTER TABLE documents ADD COLUMN document_number TEXT")
except sqlite3.OperationalError:
    pass

try:
    c.execute("ALTER TABLE original_documents ADD COLUMN full_name TEXT")
except sqlite3.OperationalError:
    pass
try:
    c.execute("ALTER TABLE original_documents ADD COLUMN date_of_birth TEXT")
except sqlite3.OperationalError:
    pass
try:
    c.execute("ALTER TABLE original_documents ADD COLUMN document_type TEXT")
except sqlite3.OperationalError:
    pass
try:
    c.execute("ALTER TABLE original_documents ADD COLUMN document_number TEXT")
except sqlite3.OperationalError:
    pass
conn.commit()
conn.close()



def store_in_db(file_content, file_extension, participant_name, document_hash, txn_hash, filename, extracted_info=None):
    if extracted_info is None:
        print("No extracted info found, falling back to empty dictionary")
        extracted_info = {}  # Fallback to empty dictionary if no info was extracted

    participant_name = extracted_info.get('full_name', '')  # Extract full name
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    
    # Extract fields from Gemini response, with fallbacks to empty string if None
    full_name = extracted_info.get("full_name", '')
    date_of_birth = extracted_info.get("date_of_birth", '')
    document_type = extracted_info.get("document_type", '')
    document_number = extracted_info.get("document_number", '')
    
    conn = sqlite3.connect('document_verification.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO documents (
            participant_name, document_hash, txn_hash, filename, timestamp,
            full_name, date_of_birth, document_type, document_number
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        participant_name, document_hash, txn_hash, filename, timestamp,
        full_name, date_of_birth, document_type, document_number
    ))
    conn.commit()
    conn.close()


    
def store_original_document(file_content, file_extension, participant_name, document_hash, content, filename, timestamp, extracted_info=None):
    participant_info = extract_participant_info(file_content, file_extension)
    participant_name = participant_info.get('full_name', '')

    full_name = extracted_info.get("full_name") if extracted_info else None
    date_of_birth = extracted_info.get("date_of_birth") if extracted_info else None
    document_type = extracted_info.get("document_type") if extracted_info else None
    document_number = extracted_info.get("document_number") if extracted_info else None
    
    conn = sqlite3.connect('document_verification.db')
    c = conn.cursor()
    # 🔒 Check if a document already exists for this participant
    c.execute('SELECT 1 FROM original_documents WHERE participant_name = ?', (participant_name,))
    if c.fetchone():
        conn.close()
        raise ValueError(f"Original document for '{participant_name}' already exists.")


    
    c.execute('''
        INSERT INTO original_documents (
            participant_name, document_hash, document_content, filename, timestamp,
            full_name, date_of_birth, document_type, document_number
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        participant_name, document_hash, content,
        filename, timestamp, full_name, date_of_birth, document_type, document_number
    ))
    conn.commit()
    conn.close()


def extract_participant_info(file_content, file_extension):
    try:
        content_b64 = base64.b64encode(file_content).decode('utf-8')
        mime_type = 'application/pdf' if file_extension == '.pdf' else 'image/jpeg'

        model = genai.GenerativeModel("gemini-1.5-flash")
        image_part = {"mime_type": mime_type, "data": content_b64}

        prompt = """
        You are given an official identification document (e.g., Aadhaar, Passport, PAN card, Driving License).
        Extract the following fields as JSON:

        {
          "full_name": "...",
          "date_of_birth": "...",
          "document_type": "...",
          "document_number": "..."
        }

        - Return valid JSON only.
        - If any field is missing or unclear, set its value to null.
        - Do not include any extra commentary.
        """

        response = model.generate_content([prompt, image_part])
        raw_text = response.text.strip()

        # DEBUG print the actual response
        print("Gemini response:", raw_text)

        # Extract JSON block from the response
        json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON object found in the model's response.")

        extracted_data = json.loads(json_match.group())

        if extracted_data.get("full_name") in [None, "", "NO_NAME_FOUND"]:
            raise ValueError("Could not extract participant name from document")

        return extracted_data

    except Exception as e:
        logger.error(f"Error extracting participant info: {str(e)}")
        raise ValueError(f"Failed to extract participant information: {str(e)}")


def handle_errors(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except ValueError as e:
            logger.error(f"Validation error: {str(e)}")
            return jsonify({"error": str(e)}), 400
        except ConnectionError as e:
            logger.error(f"Blockchain connection error: {str(e)}")
            return jsonify({"error": "Blockchain connection failed"}), 503
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return jsonify({"error": "An unexpected error occurred"}), 500
    return decorated_function
    
    

@app.route('/')
def home():
    return redirect('/role')

@app.route('/role', methods=['GET', 'POST'])
def select_role():
    if request.method == 'POST':
        role = request.form.get('role')
        if role == 'employee':
            return redirect('/upload')
        elif role == 'hr':
            return redirect('/login')
    return render_template('role_selection.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username == HR_USERNAME and password == HR_PASSWORD:
            session['hr_logged_in'] = True
            return redirect('/verify')
        else:
            flash('Invalid credentials. Try again.', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('hr_logged_in', None)
    flash('Logged out successfully.', 'success')
    return redirect('/role')

@app.route('/upload')
def upload():
    return render_template('upload.html', role='employee')


@app.route('/verify')
def verify():
    if not session.get('hr_logged_in'):
        return redirect('/login')
    return render_template('verify.html', role='hr')
    
    
@app.route('/upload_data', methods=['POST'])
@handle_errors
def upload_data():
    file = request.files.get('file')
    if not file:
        raise ValueError("No file provided")

    filename = file.filename
    file_extension = os.path.splitext(filename)[1].lower()
    if file_extension not in ['.pdf', '.jpg', '.jpeg', '.png']:
        raise ValueError("Unsupported file format")

    content = file.read()
    extracted_info = extract_participant_info(content, file_extension)
    participant_name = extracted_info.get('full_name', '')
    document_hash = hashlib.sha256(content).hexdigest()
    account = web3.eth.account.from_key(private_key)

    gas_price = int(web3.eth.gas_price * 1.1)
    transaction = {
        'from': account.address,
        'to': from_address,
        'value': web3.to_wei(0, 'ether'),
        'gas': 100000,
        'gasPrice': gas_price,
        'nonce': web3.eth.get_transaction_count(account.address),
        'data': web3.to_hex(text=document_hash),
        'chainId': chain_id
    }

    transaction['gas'] = web3.eth.estimate_gas(transaction)
    signed_txn = account.sign_transaction(transaction)
    # tx_hash = web3.eth.send_raw_transaction(signed_txn.raw_transaction)
    # tx_receipt = web3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
    # Start timing
    start_time = time.time()

    # Send transaction
    tx_hash = web3.eth.send_raw_transaction(signed_txn.raw_transaction)

    # Wait for transaction to be mined
    tx_receipt = web3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

    # End timing
    end_time = time.time()

    # Calculate latency
    latency = end_time - start_time
    transaction_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))

    # Log it
    logger.info(f"Transaction submitted at {transaction_time}")
    logger.info(f"Transaction latency: {latency:.2f} seconds")


    if tx_receipt.status == 1:
        # Here, pass the extracted content, file extension, and other details to store_in_db
        store_in_db(content, file_extension, participant_name, document_hash, tx_hash.hex(), filename, extracted_info)
        return jsonify({
            'status': 'success',
            'transaction_hash': tx_hash.hex(),
            'participant_name': participant_name,
            'document_type': extracted_info.get("document_type", "N/A"),
            'message': 'Document uploaded successfully'
        })

    else:
        raise ValueError("Blockchain transaction failed")


@app.route('/verify_document', methods=['POST'])
@handle_errors
def verify_document():
    file = request.files.get('file')
    if not file:
        raise ValueError("No file provided")

    filename = file.filename
    file_extension = os.path.splitext(filename)[1].lower()
    if file_extension not in ['.pdf', '.jpg', '.jpeg', '.png']:
        raise ValueError("Unsupported file format")

    content = file.read()
    calculated_hash = hashlib.sha256(content).hexdigest()
    extracted_info = extract_participant_info(content, file_extension)
    extracted_name = extracted_info.get('full_name', '').strip()

    # === Step 1: Fetch all original documents ===
    conn = sqlite3.connect('document_verification.db')
    c = conn.cursor()
    c.execute('''
        SELECT participant_name, full_name, date_of_birth, document_type, document_number, 
               document_hash, document_content 
        FROM original_documents
    ''')
    original_documents = c.fetchall()
    conn.close()

    matching_record = None
    field_mismatches = []

    for row in original_documents:
        (
            db_participant, db_full_name, db_dob, db_doc_type, db_doc_number,
            db_hash, db_content
        ) = row

        if extracted_name == db_full_name:  # Name matched
            if (
                extracted_info.get("date_of_birth") == db_dob and
                extracted_info.get("document_type") == db_doc_type and
                extracted_info.get("document_number") == db_doc_number
            ):
                matching_record = row
                break  # Perfect match
            else:
                if extracted_info.get("date_of_birth") != db_dob:
                    field_mismatches.append(f"DOB (Uploaded: {extracted_info.get('date_of_birth')}, Original: {db_dob})")
                if extracted_info.get("document_type") != db_doc_type:
                    field_mismatches.append(f"Type (Uploaded: {extracted_info.get('document_type')}, Original: {db_doc_type})")
                if extracted_info.get("document_number") != db_doc_number:
                    field_mismatches.append(f"Number (Uploaded: {extracted_info.get('document_number')}, Original: {db_doc_number})")
                break  # Close match but not identical

    if not matching_record:
        return jsonify({
            'status': 'error',
            'message': 'No original reference document found for this participant.',
            'extracted_name': extracted_name,
            'mismatches': field_mismatches
        }), 404

    # Unpack matched original data
    (_, _, _, _, _, original_hash, original_content) = matching_record

    # === Step 2: Check against employee-uploaded document ===
    conn = sqlite3.connect('document_verification.db')
    c = conn.cursor()
    c.execute('''
        SELECT full_name, date_of_birth, document_type, document_number, txn_hash
        FROM documents
        WHERE participant_name = ? AND document_hash = ?
        ORDER BY timestamp DESC LIMIT 1
    ''', (extracted_name, calculated_hash))
    employee_data = c.fetchone()
    conn.close()

    if not employee_data:
        return jsonify({
            'status': 'error',
            'message': 'No matching employee-uploaded document found.',
            'extracted_name': extracted_name
        }), 404

    emp_full_name, emp_dob, emp_type, emp_number, txn_hash = employee_data

    employee_mismatches = []
    if extracted_info.get('full_name') != emp_full_name:
        employee_mismatches.append(f"Full Name (Uploaded: {extracted_info.get('full_name')}, Employee: {emp_full_name})")
    if extracted_info.get('date_of_birth') != emp_dob:
        employee_mismatches.append(f"DOB (Uploaded: {extracted_info.get('date_of_birth')}, Employee: {emp_dob})")
    if extracted_info.get('document_type') != emp_type:
        employee_mismatches.append(f"Type (Uploaded: {extracted_info.get('document_type')}, Employee: {emp_type})")
    if extracted_info.get('document_number') != emp_number:
        employee_mismatches.append(f"Number (Uploaded: {extracted_info.get('document_number')}, Employee: {emp_number})")

    if employee_mismatches:
        return jsonify({
            'status': 'error',
            'message': f'❌ Document fields mismatched: {", ".join(employee_mismatches)}',
            'extracted_name': extracted_name
        }), 400

    # === Step 3: Check content match ===
    if calculated_hash != original_hash or content != original_content:
        return jsonify({
            'status': 'error',
            'message': 'Uploaded document content does not match original reference file.',
            'extracted_name': extracted_name,
            'mismatches': field_mismatches
        }), 400

    # === Step 4: Blockchain verification ===
    try:
        receipt = web3.eth.get_transaction_receipt(txn_hash)
        if receipt and receipt['status'] == 1:
            return jsonify({
                'status': 'success',
                'message': '✅ Document verified! It matches original and has a valid blockchain record.',
                'transaction_hash': txn_hash,
                'extracted_name': extracted_name,
                'stored_hash': calculated_hash
            })
        else:
            return jsonify({
                'status': 'warning',
                'message': '⚠️ Matches original and employee upload, but blockchain transaction failed.',
                'transaction_hash': txn_hash,
                'extracted_name': extracted_name
            })
    except Exception as e:
        return jsonify({
            'status': 'warning',
            'message': '⚠️ Matches original and upload, but blockchain lookup failed.',
            'extracted_name': extracted_name
        })



@app.route('/admin_upload', methods=['GET', 'POST'])
def admin_upload():
    if not session.get('hr_logged_in'):
        return redirect('/login')

    if request.method == 'POST':
        file = request.files.get('file')
        if not file:
            flash('No file uploaded.', 'danger')
            return redirect('/admin_upload')

        filename = file.filename
        file_extension = os.path.splitext(filename)[1].lower()
        if file_extension not in ['.pdf', '.jpg', '.jpeg', '.png']:
            flash('Unsupported file format.', 'danger')
            return redirect('/admin_upload')

        try:
            content = file.read()
            document_hash = hashlib.sha256(content).hexdigest()
            extracted_info = extract_participant_info(content, file_extension)  # should return dict
            participant_name = extracted_info.get("full_name", "")
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

            store_original_document(
                file_content=content,
                file_extension=file_extension,
                participant_name=participant_name,
                document_hash=document_hash,
                content=content,
                filename=filename,
                timestamp=timestamp,
                extracted_info=extracted_info
            )

            flash(f'Document for {participant_name} uploaded successfully.', 'success')
        except Exception as e:
            logger.error(f"Admin upload failed: {str(e)}")
            flash(f'Error: {str(e)}', 'danger')

        return redirect('/admin_upload')

    return render_template('admin_upload.html', role='hr')



@app.route('/admin_originals')
def admin_originals():
    if not session.get('hr_logged_in'):
        return redirect('/login')

    conn = sqlite3.connect('document_verification.db')
    c = conn.cursor()
    c.execute('''
        SELECT o.participant_name, o.document_hash, o.filename, o.timestamp,
               CASE 
                   WHEN d.txn_hash IS NOT NULL AND d.txn_hash != '' THEN 'Verified'
                   ELSE 'Not Verified'
               END as verification_status
        FROM original_documents o
        LEFT JOIN (
            SELECT document_hash, txn_hash
            FROM documents
            GROUP BY document_hash
            HAVING MAX(timestamp)
        ) d ON o.document_hash = d.document_hash
    ''')
    rows = c.fetchall()
    conn.close()

    return render_template('original_list.html', documents=rows, role='hr')

from flask import Response, send_file
import io
import mimetypes

@app.route('/download_original/<participant_name>')
def download_original(participant_name):
    if not participant_name.strip():
        return "Invalid participant name", 400

    conn = sqlite3.connect('document_verification.db')
    cursor = conn.cursor()
    cursor.execute("SELECT document_content, filename FROM original_documents WHERE participant_name = ? ORDER BY timestamp DESC LIMIT 1", (participant_name,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return "Document not found", 404

    content, filename = row
    mimetype, _ = mimetypes.guess_type(filename)
    if mimetype is None:
        mimetype = 'application/octet-stream'

    return send_file(
        io.BytesIO(content),
        as_attachment=True,
        download_name=filename or f"{participant_name}_original_document",
        mimetype=mimetype
    )

@app.route('/delete_original/<participant_name>', methods=['POST'])
def delete_original(participant_name):
    if not session.get('hr_logged_in'):
        return redirect('/login')

    conn = sqlite3.connect('document_verification.db')
    c = conn.cursor()
    c.execute("DELETE FROM original_documents WHERE participant_name = ?", (participant_name,))
    conn.commit()
    conn.close()

    flash(f"Document for {participant_name} deleted successfully.", "success")
    return redirect('/admin_originals')


if __name__ == '__main__':
    app.run(debug=True)