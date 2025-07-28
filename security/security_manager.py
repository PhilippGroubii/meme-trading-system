"""
Security Manager for Trading System
Encrypts API keys, trading data, and communications
"""

import os
import json
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import hashlib
from typing import Dict, Any

class SecurityManager:
    def __init__(self, master_password: str = None):
        """Initialize security manager with encryption"""
        self.master_password = master_password or os.getenv('MASTER_PASSWORD', 'change_me_now')
        self.key = self._derive_key(self.master_password)
        self.cipher = Fernet(self.key)
        
    def _derive_key(self, password: str) -> bytes:
        """Derive encryption key from password"""
        salt = b'trading_system_salt_2024'  # In production, use random salt
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key
    
    def encrypt_data(self, data: Any) -> str:
        """Encrypt any data (API keys, trade data, etc.)"""
        json_data = json.dumps(data)
        encrypted = self.cipher.encrypt(json_data.encode())
        return base64.urlsafe_b64encode(encrypted).decode()
    
    def decrypt_data(self, encrypted_data: str) -> Any:
        """Decrypt data back to original format"""
        encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
        decrypted = self.cipher.decrypt(encrypted_bytes)
        return json.loads(decrypted.decode())
    
    def secure_api_keys(self, api_keys: Dict[str, str]) -> str:
        """Encrypt API keys for secure storage"""
        return self.encrypt_data(api_keys)
    
    def load_api_keys(self, encrypted_keys: str) -> Dict[str, str]:
        """Load and decrypt API keys"""
        return self.decrypt_data(encrypted_keys)
    
    def hash_portfolio_data(self, portfolio: Dict) -> str:
        """Create tamper-proof hash of portfolio data"""
        portfolio_str = json.dumps(portfolio, sort_keys=True)
        return hashlib.sha256(portfolio_str.encode()).hexdigest()
    
    def validate_portfolio_integrity(self, portfolio: Dict, expected_hash: str) -> bool:
        """Validate portfolio data hasn't been tampered with"""
        current_hash = self.hash_portfolio_data(portfolio)
        return current_hash == expected_hash
    
    def secure_trade_log(self, trade_data: Dict) -> Dict:
        """Create tamper-proof trade log entry"""
        trade_data['timestamp'] = str(trade_data.get('timestamp', ''))
        trade_data['signature'] = hashlib.sha256(
            json.dumps(trade_data, sort_keys=True).encode()
        ).hexdigest()
        return trade_data

# Usage example
if __name__ == "__main__":
    # Initialize security manager
    security = SecurityManager("your_strong_master_password_here")
    
    # Test API key encryption
    api_keys = {
        'REDDIT_CLIENT_ID': 'your_reddit_id',
        'REDDIT_CLIENT_SECRET': 'your_reddit_secret',
        'COINGECKO_API_KEY': 'your_coingecko_key'
    }
    
    # Encrypt
    encrypted_keys = security.secure_api_keys(api_keys)
    print("✅ API keys encrypted")
    
    # Decrypt
    decrypted_keys = security.load_api_keys(encrypted_keys)
    print("✅ API keys decrypted successfully")
    
    # Test portfolio integrity
    portfolio = {'cash': 10000, 'positions': {'DOGE': 1000}}
    portfolio_hash = security.hash_portfolio_data(portfolio)
    is_valid = security.validate_portfolio_integrity(portfolio, portfolio_hash)
    print(f"✅ Portfolio integrity: {is_valid}")