import sys
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.fernet import Fernet
import os
from dotenv import load_dotenv


load_dotenv()

def decrypt_file_content(encrypted_file_path):
    private_key_pem = os.environ.get("SUBMISSION_PRIVATE_KEY")
    """
    Decrypts a hybrid-encrypted file (RSA + Fernet).
    
    Args:
        encrypted_file_path (str): Path to the .enc file
        private_key_pem (str): The RSA Private Key string
        
    Returns:
        bytes: The raw decrypted content of the file
    """
    try:
        private_key = serialization.load_pem_private_key(
            private_key_pem.encode('utf-8'),
            password=None
        )
    except Exception as e:
        raise ValueError(f"Invalid Private Key format: {e}")

    try:
        with open(encrypted_file_path, "rb") as f:
            file_content = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {encrypted_file_path}")

    rsa_segment_size = 256 
    
    if len(file_content) < rsa_segment_size:
        raise ValueError("File is too short to contain a valid encrypted header.")

    encrypted_session_key = file_content[:rsa_segment_size]
    encrypted_data = file_content[rsa_segment_size:]

    try:
        session_key = private_key.decrypt(
            encrypted_session_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
    except Exception as e:
        raise ValueError(f"RSA Decryption failed (Wrong Key?): {e}")

    try:
        cipher_suite = Fernet(session_key)
        decrypted_data = cipher_suite.decrypt(encrypted_data)
        return decrypted_data
    except Exception as e:
        raise ValueError(f"Data Decryption failed (Corrupted data?): {e}")
    
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python decrypt.py <filename>")
    else:
        file_content = decrypt_file_content(sys.argv[1])
        new_file_name = sys.argv[1].replace(".enc", "")
        with open(new_file_name, "wb") as f:
            f.write(file_content)
        print(f"Decryption successful! Decrypted file saved as '{new_file_name}'")